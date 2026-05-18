import json
import logging
import re
import time
import asyncio
from typing import Dict, Any, Optional, Union

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
import base64
from voice_processor import VoiceProcessor
from connection_manager import ConnectionManager
from config import settings

SENTENCE_ENDINGS = re.compile(r'[。！？!?…]+')
ASR_TAG_PATTERN = re.compile(r'<\|[^|]+\|>')
from live2d_prompt import (
    allowed_emotions_frozenset,
    clean_llm_reply_for_history,
    parse_current_live2d_model_key_from_settings,
    sanitize_emotion_tags,
    strip_emotion_tags as strip_emotion_tags_for_tts,
)
from minimax_emotion_interjection import apply_minimax_emotion_interjection

MIN_SPEECH_BYTES = 3200   # 最少 200ms 的 16kHz int16 音频才送 ASR

logger = logging.getLogger(__name__)


def _pcm_int16_bytes_to_f32(audio_data: bytes) -> Optional[np.ndarray]:
    """将客户端 PCM int16 小端字节转为 float32 波形 [-1,1]，供 FunASR 走 ndarray 分支。"""
    if not audio_data or len(audio_data) < 2 or (len(audio_data) % 2) != 0:
        return None
    pcm = np.frombuffer(audio_data, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0

# 全局变量（将在main.py中注入）
voice_processor: VoiceProcessor = None
manager: ConnectionManager = None

def init_websocket_routes(app, voice_proc: VoiceProcessor, conn_manager: ConnectionManager):
    """初始化WebSocket路由"""
    global voice_processor, manager
    voice_processor = voice_proc
    manager = conn_manager
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket端点"""
        try:
            # 检查服务器是否正在关闭
            if manager._shutting_down:
                await websocket.close(code=1000, reason="服务器正在关闭")
                return
                
            print("WebSocket连接尝试")
            client_id = str(id(websocket))
            await manager.connect(websocket, client_id)

            _kw = (getattr(settings, 'WAKEUP_KEYWORD', '') or '').strip()
            if _kw:
                manager.set_wakeup_state(client_id, 'sleep')
                logger.info(f"客户端 {client_id} 唤醒词已设置({_kw})，初始进入休眠状态")
            else:
                manager.set_wakeup_state(client_id, 'always_on')
                logger.info(f"客户端 {client_id} 未设置唤醒词，始终监听模式")
            
            try:
                while True:
                    # 检查服务器是否正在关闭
                    if manager._shutting_down:
                        break
                        
                    # 接收数据，可以是文本或二进制
                    try:
                        # 使用通用接收方法
                        message = await websocket.receive()
                        
                        # 检查连接断开
                        if message["type"] == "websocket.disconnect":
                            break
                            
                        # 根据消息类型处理
                        if message["type"] == "websocket.receive":
                            if "text" in message:
                                # 处理文本消息（控制命令等）
                                data = message["text"]
                                text_message = json.loads(data)
                                
                                # 处理控制命令
                                if text_message['type'] == 'control':
                                    await handle_control_command(text_message, client_id, websocket)
                                
                                # 处理中断信号
                                elif text_message['type'] == 'interrupt':
                                    await manager.set_interrupt_flag(client_id, True)
                                    await manager.cancel_current_tts(client_id)
                                    await websocket.send_json({
                                        'type': 'interrupt_ack',
                                        'message': '中断信号已接收'
                                    })
                            
                            elif "bytes" in message:
                                # 处理二进制数据（音频数据）
                                audio_data = message["bytes"]
                                await handle_audio_data(audio_data, client_id, websocket)
                        
                    except WebSocketDisconnect:
                        logger.info(f"客户端 {client_id} 断开连接")
                        break
                    except Exception as e:
                        logger.error(f"处理WebSocket消息错误: {e}")
                        await websocket.send_json({
                            'type': 'error',
                            'message': f'消息处理错误: {str(e)}',
                            'timestamp': time.time()
                        })
                        continue
            
            except Exception as e:
                logger.error(f"WebSocket连接处理错误: {e}")
            finally:
                manager.disconnect(client_id)
                logger.info(f"客户端 {client_id} 连接已清理")
        
        except Exception as e:
            logger.error(f"WebSocket连接建立错误: {e}")

async def handle_control_command(text_message: Dict[str, Any], client_id: str, websocket: WebSocket):
    """处理控制命令"""
    command = text_message.get('command')
    
    if command == 'start_listening':
        manager.user_data[client_id]['is_listening'] = True
        await websocket.send_json({
            'type': 'listening_started',
            'message': '开始监听',
            'timestamp': time.time()
        })
    
    elif command == 'stop_listening':
        manager.user_data[client_id]['is_listening'] = False
        await websocket.send_json({
            'type': 'listening_stopped',
            'message': '停止监听',
            'timestamp': time.time()
        })
    
    elif command == 'enable_interrupt':
        manager.user_data[client_id]['interrupt_enabled'] = True
        await websocket.send_json({
            'type': 'interrupt_enabled',
            'message': '语音打断功能已启用'
        })
    
    elif command == 'disable_interrupt':
        manager.user_data[client_id]['interrupt_enabled'] = False
        await websocket.send_json({
            'type': 'interrupt_disabled',
            'message': '语音打断功能已禁用',
            'timestamp': time.time()
        })
    
    elif command == 'delete_message':
        message_index = text_message.get('index')
        conversation_history = manager.user_data[client_id].get('conversation_history', [])
        
        if message_index is not None and 0 <= message_index < len(conversation_history):
            deleted_msg = conversation_history.pop(message_index)
            logger.info(f"删除消息 - 客户端 {client_id}: {deleted_msg['type']} - {deleted_msg['text'][:30]}...")
            await websocket.send_json({
                'type': 'message_deleted',
                'index': message_index,
                'message': '消息删除成功'
            })
        else:
            await websocket.send_json({
                'type': 'error',
                'message': '消息索引无效',
                'timestamp': time.time()
            })
    
    elif command == 'clear_conversation':
        manager.user_data[client_id]['conversation_history'] = []
        logger.info(f"清空对话历史 - 客户端 {client_id}")
        await websocket.send_json({
            'type': 'conversation_cleared',
            'message': '对话历史已清空',
            'timestamp': time.time()
        })

    elif command == 'voiceprint_record':
        manager.user_data[client_id]['voiceprint_recording'] = True
        await websocket.send_json({
            'type': 'voiceprint_recording_started',
            'message': '请开始说话，系统将录制声纹',
            'timestamp': time.time()
        })

    elif command == 'voiceprint_clear':
        manager.clear_voiceprint(client_id)
        await websocket.send_json({
            'type': 'voiceprint_cleared',
            'message': '声纹已清除',
            'timestamp': time.time()
        })

def create_wav_header(audio_bytes: bytes, sample_rate: int = 24000, num_channels: int = 1, bits_per_sample: int = 16):
    """创建WAV文件头"""
    import struct
    
    import wave
    import io
    
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(bits_per_sample // 8)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)
    
    return buffer.getvalue()

async def handle_tts_request(
    sentence_raw: str,
    client_id: str,
    websocket: WebSocket,
    allowed_emotions,
    live2d_model_key: Optional[str] = None,
):
    """对 LLM 片段做标签白名单清洗；TTS 用净文本；前端用带合法标签的副本驱动 Live2D。"""
    try:
        cleaned_tags = sanitize_emotion_tags(sentence_raw or "", allowed_emotions)
        tts_plain = strip_emotion_tags_for_tts(cleaned_tags)
        if not tts_plain:
            return

        tts_model = getattr(settings, 'TTS_MODEL', 'EdgeTTS')

        if tts_model == 'EdgeTTS':
            await handle_edge_tts_request(tts_plain, websocket, cleaned_tags)
        elif tts_model == 'MiniMax':
            tts_minimax = apply_minimax_emotion_interjection(
                tts_plain, cleaned_tags, live2d_model_key, settings
            )
            await handle_minimax_tts_request(tts_minimax, websocket, cleaned_tags)

    except Exception as e:
        logger.error(f"处理TTS请求错误: {e}")
        plain = strip_emotion_tags_for_tts(sanitize_emotion_tags(sentence_raw or "", allowed_emotions))
        await websocket.send_json({
            'type': 'tts_error',
            'text': plain,
            'message': f'TTS处理错误: {str(e)}',
            'timestamp': time.time()
        })

async def handle_minimax_tts_request(sentence_plain: str, websocket: WebSocket, text_with_tags: str):
    """处理 MiniMax 云端 TTS（返回 WAV 二进制，与 Edge 路径一致送前端 base64）"""
    try:
        if not voice_processor:
            await websocket.send_json({
                'type': 'tts_error',
                'message': 'VoiceProcessor未初始化',
                'timestamp': time.time()
            })
            return

        if not getattr(settings, 'MINIMAX_API_KEY', '').strip():
            await websocket.send_json({
                'type': 'tts_error',
                'message': '未配置 MINIMAX_API_KEY，请在环境变量或 .env 中设置',
                'timestamp': time.time()
            })
            return

        wav_data = await voice_processor.generate_minimax_tts(sentence_plain)

        if wav_data:
            audio_base64 = base64.b64encode(wav_data).decode('utf-8')
            await websocket.send_json({
                'type': 'tts_audio',
                'audio': audio_base64,
                'text': sentence_plain,
                'textWithTags': text_with_tags,
                'message': 'MiniMax TTS生成成功',
                'timestamp': time.time()
            })
        else:
            await websocket.send_json({
                'type': 'tts_error',
                'message': 'MiniMax TTS生成失败'
            })

    except Exception as e:
        logger.error(f"MiniMax TTS处理错误: {e}")
        await websocket.send_json({
            'type': 'tts_error',
            'text': sentence_plain,
            'message': f'MiniMax TTS处理错误: {str(e)}',
            'timestamp': time.time()
        })

async def handle_edge_tts_request(sentence_plain: str, websocket: WebSocket, text_with_tags: str):
    """处理 EdgeTTS 请求"""
    try:
        if not voice_processor:
            await websocket.send_json({
                'type': 'tts_error',
                'message': 'VoiceProcessor未初始化',
                'timestamp': time.time()
            })
            return
        
        wav_data = await voice_processor.generate_edge_tts(sentence_plain)
        
        if wav_data:
            audio_base64 = base64.b64encode(wav_data).decode('utf-8')
            await websocket.send_json({
                'type': 'tts_audio',
                'audio': audio_base64,
                'text': sentence_plain,
                'textWithTags': text_with_tags,
                'message': 'EdgeTTS音频生成成功',
                'timestamp': time.time()
            })
        else:
            await websocket.send_json({
                'type': 'tts_error',
                'message': 'EdgeTTS音频生成失败'
            })
    
    except Exception as e:
        logger.error(f"EdgeTTS处理错误: {e}")
        await websocket.send_json({
            'type': 'tts_error',
            'text': sentence_plain,
            'message': f'EdgeTTS处理错误: {str(e)}',
            'timestamp': time.time()
        })

async def handle_audio_data(audio_data: bytes, client_id: str, websocket: WebSocket):
    """接收音频块，积累语音段，静音结束后触发 ASR → LLM → TTS 管道"""
    try:
        user_data = manager.user_data[client_id]
        current_time = time.time()

        if user_data.get('voiceprint_recording'):
            if 'voiceprint_audio' not in user_data:
                user_data['voiceprint_audio'] = []
            user_data['voiceprint_audio'].append(audio_data)
            total_bytes = sum(len(c) for c in user_data['voiceprint_audio'])
            if total_bytes >= 16000 * 2 * 3:
                all_audio = b''.join(user_data['voiceprint_audio'])
                user_data['voiceprint_audio'] = []
                user_data['voiceprint_recording'] = False
                if voice_processor and voice_processor.voiceprint_model:
                    embedding = await voice_processor.extract_voiceprint_feature(all_audio)
                    if embedding is not None:
                        manager.set_voiceprint_embedding(client_id, embedding)
                        await websocket.send_json({
                            'type': 'voiceprint_recorded',
                            'message': '声纹录制成功',
                            'timestamp': time.time()
                        })
                        return
                await websocket.send_json({
                    'type': 'voiceprint_record_failed',
                    'message': '声纹录制失败，请重试',
                    'timestamp': time.time()
                })
            return

        if voice_processor:
            vad_result = voice_processor.detect_speech(audio_data, client_id)
        else:
            # VAD 不可用时直接积累所有音频，60 秒超时兜底
            user_data['speech_buffer'].append(audio_data)
            user_data['last_speech_time'] = current_time
            user_data['is_speech_active'] = True
            manager.update_activity(client_id)
            return

        if vad_result.has_speech:
            user_data['speech_buffer'].append(audio_data)
            user_data['last_speech_time'] = current_time
            if not user_data['is_speech_active']:
                user_data['is_speech_active'] = True
                logger.debug(f"客户端 {client_id} 检测到语音开始")
        elif user_data['is_speech_active']:
            silence_duration = current_time - user_data['last_speech_time']
            if silence_duration >= getattr(settings, "speech_end_silence_sec", 0.55):
                user_data['is_speech_active'] = False
                buffered_audio = b''.join(user_data['speech_buffer'])
                user_data['speech_buffer'] = []
                logger.debug(f"客户端 {client_id} 语音段结束，积累 {len(buffered_audio)} 字节")
                if len(buffered_audio) >= MIN_SPEECH_BYTES:
                    if not user_data.get('is_processing', False):
                        _timeout = getattr(settings, 'WAKEUP_TIMEOUT', 60)
                        if manager.check_wakeup_timeout(client_id, _timeout):
                            manager.set_wakeup_state(client_id, 'sleep')
                            logger.info(f"客户端 {client_id} 唤醒超时({_timeout}s)，进入休眠")
                            await websocket.send_json({
                                'type': 'wakeup_sleep',
                                'message': f'已{_timeout}秒无对话，进入休眠',
                                'timestamp': time.time()
                            })
                        user_data['is_processing'] = True
                        asyncio.create_task(
                            _run_speech_pipeline(buffered_audio, client_id, websocket)
                        )
                    else:
                        logger.debug(f"客户端 {client_id} 上一段语音仍在处理，跳过本段")

        manager.update_activity(client_id)

    except Exception as e:
        logger.error(f"处理音频数据错误 - 客户端 {client_id}: {e}")


async def _run_speech_pipeline(audio_data: bytes, client_id: str, websocket: WebSocket):
    """包装 process_speech，确保无论是否异常都会清除 is_processing 标志"""
    try:
        await process_speech(audio_data, client_id, websocket)
    finally:
        if client_id in manager.user_data:
            manager.user_data[client_id]['is_processing'] = False


async def tts_worker(
    queue: asyncio.Queue,
    client_id: str,
    websocket: WebSocket,
    allowed_emotions,
    live2d_model_key: Optional[str] = None,
):
    """从队列中依次取句子生成并发送 TTS 音频"""
    while True:
        sentence = await queue.get()
        if sentence is None:
            break
        try:
            await handle_tts_request(
                sentence, client_id, websocket, allowed_emotions, live2d_model_key
            )
        except Exception as e:
            logger.error(f"TTS worker 错误 - 客户端 {client_id}: {e}")


async def process_speech(audio_data: bytes, client_id: str, websocket: WebSocket):
    """对完整语音段执行 ASR → LLM（流式）→ TTS（并行）管道"""
    if not voice_processor or not voice_processor.asr_model:
        logger.warning("ASR模型未初始化")
        return

    try:
        waveform = _pcm_int16_bytes_to_f32(audio_data)
        if waveform is None:
            logger.warning("ASR: PCM 字节无效（长度或对齐） client=%s", client_id)
            return
        _in: Union[np.ndarray, bytes] = waveform
        _gen = {"input": _in}
        _lang = (getattr(settings, "asr_language", None) or "auto").strip().lower()
        if _lang and _lang != "auto":
            _gen["language"] = _lang
        logger.debug(
            "ASR generate kwargs: language=%s, samples=%d",
            _gen.get("language", "auto"),
            len(waveform),
        )
        try:
            result = voice_processor.asr_model.generate(**_gen)
        except TypeError:
            # 非 SenseVoice 等不接收 language 参数时回退
            if len(_gen) > 1:
                result = voice_processor.asr_model.generate(input=_in)
            else:
                raise
        if not (result and len(result) > 0 and 'text' in result[0]):
            return

        raw_text = result[0]['text'].strip()
        recognized_text = ASR_TAG_PATTERN.sub('', raw_text).strip()
        if not recognized_text:
            return

        logger.info(f"ASR识别 - 客户端 {client_id}: {recognized_text}")

        _wakeup_state = manager.get_wakeup_state(client_id)
        _wakeup_kw = (getattr(settings, 'WAKEUP_KEYWORD', '') or '').strip()
        _wakeup_matched = False

        if _wakeup_state == 'sleep' and _wakeup_kw:
            if voice_processor.check_wakeup_keyword(recognized_text, _wakeup_kw):
                _wakeup_matched = True
                manager.set_wakeup_state(client_id, 'awake')
                await websocket.send_json({
                    'type': 'wakeup_detected',
                    'message': f'唤醒词「{_wakeup_kw}」已检测',
                    'timestamp': time.time()
                })
                original_wakeup_text = recognized_text
                recognized_text = voice_processor.strip_wakeup_keyword(recognized_text, _wakeup_kw)
                logger.info(f"客户端 {client_id} 唤醒成功，处理文本: {recognized_text}")
                conversation_history = manager.user_data[client_id]['conversation_history']
                conversation_history.append({
                    'type': 'user',
                    'text': original_wakeup_text,
                    'timestamp': time.time()
                })
                await websocket.send_json({
                    'type': 'llm_stream_start',
                    'text': original_wakeup_text,
                    'role': 'user',
                    'timestamp': time.time()
                })
                if not recognized_text:
                    _greeting = '你好！我在，有什么可以帮你的？'
                    conversation_history.append({'type': 'assistant', 'text': _greeting, 'timestamp': time.time()})
                    live_key = parse_current_live2d_model_key_from_settings(settings)
                    allowed = allowed_emotions_frozenset(live_key)
                    await handle_tts_request(_greeting, client_id, websocket, allowed, live_key)
                    await websocket.send_json({
                        'type': 'llm_stream_end',
                        'role': 'assistant',
                        'timestamp': time.time()
                    })
                    return
            else:
                logger.info(f"客户端 {client_id} 休眠中，未检测到唤醒词「{_wakeup_kw}」，忽略: {recognized_text}")
                await websocket.send_json({
                    'type': 'wakeup_required',
                    'message': f'请先说唤醒词「{_wakeup_kw}」来激活对话',
                    'timestamp': time.time()
                })
                return
        elif _wakeup_state == 'awake':
            manager.update_wakeup_activity(client_id)

        if not _wakeup_matched:
            conversation_history = manager.user_data[client_id]['conversation_history']
            conversation_history.append({
                'type': 'user',
                'text': recognized_text,
                'timestamp': time.time()
            })

        if manager.has_voiceprint(client_id):
            if not voice_processor or not voice_processor.voiceprint_model:
                logger.error(f"客户端 {client_id} 声纹验证已开启但模型未就绪，拒绝处理")
                await websocket.send_json({
                    'type': 'voiceprint_rejected',
                    'message': '声纹模型未就绪，请检查 modelscope/campplus 是否安装',
                    'timestamp': time.time()
                })
                return

            audio_duration = len(audio_data) / 2 / 16000.0
            stored_embedding = manager.get_voiceprint_embedding()
            if stored_embedding is None:
                logger.warning(f"客户端 {client_id} 声纹嵌入未提取（ffmpeg缺失？），拒绝处理")
                await websocket.send_json({
                    'type': 'voiceprint_rejected',
                    'message': '声纹嵌入未就绪，请重新录制并确保 ffmpeg 已安装',
                    'timestamp': time.time()
                })
                return
            if audio_duration >= 1.0:
                current_embedding = await voice_processor.extract_voiceprint_feature(audio_data)
                if current_embedding is None:
                    logger.warning(f"客户端 {client_id} 声纹提取失败（音频过短或模型异常），拒绝处理")
                    await websocket.send_json({
                        'type': 'voiceprint_rejected',
                        'message': '声纹提取失败，请重试',
                        'timestamp': time.time()
                    })
                    return
                if not voice_processor.match_voiceprint(current_embedding, stored_embedding):
                    logger.warning(f"客户端 {client_id} 声纹不匹配，拒绝处理")
                    await websocket.send_json({
                        'type': 'voiceprint_rejected',
                        'message': '声纹不匹配，仅授权用户可使用',
                        'timestamp': time.time()
                    })
                    return

        if not (voice_processor and voice_processor.llm_client):
            await websocket.send_json({
                'type': 'llm_unavailable',
                'message': 'LLM服务暂不可用',
                'timestamp': time.time()
            })
            return

        live_key = parse_current_live2d_model_key_from_settings(settings)
        allowed = allowed_emotions_frozenset(live_key)
        logger.info(
            "Live2D→LLM: model_key=%s, 可用情绪种类=%s",
            live_key or "(未绑定)",
            len(allowed),
        )

        if not _wakeup_matched:
            await websocket.send_json({
                'type': 'llm_stream_start',
                'text': recognized_text,
                'role': 'user',
                'timestamp': time.time()
            })

        tts_queue: asyncio.Queue = asyncio.Queue()
        tts_task = asyncio.create_task(
            tts_worker(tts_queue, client_id, websocket, allowed, live_key)
        )

        llm_response = ""
        sentence_buffer = ""
        _tts_choice = getattr(settings, 'TTS_MODEL', 'EdgeTTS')
        if _tts_choice == 'EdgeTTS':
            tts_enabled = True
        elif _tts_choice == 'MiniMax':
            tts_enabled = bool(getattr(settings, 'MINIMAX_API_KEY', '').strip())
        else:
            tts_enabled = True

        llm_stream_ok = False
        try:
            async for chunk in voice_processor.stream_llm_response(
                recognized_text, conversation_history, live2d_model_key=live_key
            ):
                llm_response += chunk
                sentence_buffer += chunk
                while SENTENCE_ENDINGS.search(sentence_buffer):
                    match = SENTENCE_ENDINGS.search(sentence_buffer)
                    sentence = sentence_buffer[:match.end()].strip()
                    sentence_buffer = sentence_buffer[match.end():]
                    if sentence and tts_enabled:
                        await tts_queue.put(sentence)

            if sentence_buffer.strip() and tts_enabled:
                await tts_queue.put(sentence_buffer.strip())

            await tts_queue.put(None)

            await websocket.send_json({
                'type': 'llm_stream_end',
                'role': 'assistant',
                'timestamp': time.time()
            })

            await tts_task
            llm_stream_ok = True
        finally:
            if not tts_task.done():
                try:
                    await tts_queue.put(None)
                except Exception:
                    pass
                try:
                    await tts_task
                except Exception:
                    pass

        if llm_stream_ok:
            hist_text = clean_llm_reply_for_history(llm_response, allowed)
            conversation_history.append({
                'type': 'assistant',
                'text': hist_text,
                'timestamp': time.time()
            })
            logger.info(f"LLM回复(入库已剥标签) - 客户端 {client_id}: {hist_text[:60]}...")

    except Exception as e:
        logger.error(f"语音处理管道错误 - 客户端 {client_id}: {e}")
        await websocket.send_json({
            'type': 'llm_error',
            'message': f'处理失败: {str(e)}'
        })