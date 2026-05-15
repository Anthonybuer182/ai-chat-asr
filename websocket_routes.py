import json
import logging
import re
import time
import asyncio
from typing import Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
import base64
import ChatTTS
from voice_processor import VoiceProcessor
from connection_manager import ConnectionManager
from config import settings

SENTENCE_ENDINGS = re.compile(r'[。！？!?…]+')
ASR_TAG_PATTERN = re.compile(r'<\|[^|]+\|>')
# Live2D 控制标签，与 voice_processor 中提示一致；TTS 无特殊语义，若不剔除会整段朗读
LIVE2D_EMOTION_TAG_PATTERN = re.compile(r'\[EMOTION:\w+\]', re.IGNORECASE)
SILENCE_THRESHOLD = 0.8   # 静音多少秒后触发 ASR
MIN_SPEECH_BYTES = 3200   # 最少 200ms 的 16kHz int16 音频才送 ASR

logger = logging.getLogger(__name__)

def strip_emotion_tags_for_tts(text: str) -> str:
    """去掉情绪控制标签后再送 TTS，避免朗读方括号与英文词。"""
    if not text:
        return ''
    return LIVE2D_EMOTION_TAG_PATTERN.sub('', text).strip()

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

async def handle_tts_request(sentence: str, client_id: str, websocket: WebSocket):
    """处理TTS请求"""
    try:
        sentence = strip_emotion_tags_for_tts(sentence)
        if not sentence:
            return

        tts_model = getattr(settings, 'TTS_MODEL', 'ChatTTS')

        if tts_model == 'EdgeTTS':
            await handle_edge_tts_request(sentence, websocket)
        elif tts_model == 'MiniMax':
            await handle_minimax_tts_request(sentence, websocket)
        else:
            await handle_chat_tts_request(sentence, websocket)
    
    except Exception as e:
        logger.error(f"处理TTS请求错误: {e}")
        await websocket.send_json({
            'type': 'tts_error',
            'text': sentence,
            'message': f'TTS处理错误: {str(e)}',
            'timestamp': time.time()
        })

async def handle_chat_tts_request(sentence: str, websocket: WebSocket):
    """处理 ChatTTS 请求"""
    try:
        if not voice_processor or not voice_processor.tts_model:
            await websocket.send_json({
                'type': 'tts_error',
                'message': 'ChatTTS模型未初始化',
                'timestamp': time.time()
            })
            return
        
        texts = [sentence]
        
        params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=voice_processor.rand_spk,
            temperature=0.3
        )
        
        wavs = voice_processor.tts_model.infer(
            texts, 
            params_infer_code=params_infer_code,
            use_decoder=True
        )
        
        if wavs and len(wavs) > 0:
            audio_data = wavs[0]
            
            import numpy as np
            audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
            wav_data = create_wav_header(audio_bytes, sample_rate=24000)
            audio_base64 = base64.b64encode(wav_data).decode('utf-8')
            
            await websocket.send_json({
                'type': 'tts_audio',
                'audio': audio_base64,
                'text': sentence,
                'message': 'ChatTTS音频生成成功',
                'timestamp': time.time()
            })
        else:
            await websocket.send_json({
                'type': 'tts_error',
                'message': 'ChatTTS音频生成失败'
            })
    
    except Exception as e:
        logger.error(f"ChatTTS处理错误: {e}")
        await websocket.send_json({
            'type': 'tts_error',
            'text': sentence,
            'message': f'ChatTTS处理错误: {str(e)}',
            'timestamp': time.time()
        })

async def handle_minimax_tts_request(sentence: str, websocket: WebSocket):
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

        wav_data = await voice_processor.generate_minimax_tts(sentence)

        if wav_data:
            audio_base64 = base64.b64encode(wav_data).decode('utf-8')
            await websocket.send_json({
                'type': 'tts_audio',
                'audio': audio_base64,
                'text': sentence,
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
            'text': sentence,
            'message': f'MiniMax TTS处理错误: {str(e)}',
            'timestamp': time.time()
        })

async def handle_edge_tts_request(sentence: str, websocket: WebSocket):
    """处理 EdgeTTS 请求"""
    try:
        if not voice_processor:
            await websocket.send_json({
                'type': 'tts_error',
                'message': 'VoiceProcessor未初始化',
                'timestamp': time.time()
            })
            return
        
        wav_data = await voice_processor.generate_edge_tts(sentence)
        
        if wav_data:
            audio_base64 = base64.b64encode(wav_data).decode('utf-8')
            await websocket.send_json({
                'type': 'tts_audio',
                'audio': audio_base64,
                'text': sentence,
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
            'text': sentence,
            'message': f'EdgeTTS处理错误: {str(e)}',
            'timestamp': time.time()
        })

async def handle_audio_data(audio_data: bytes, client_id: str, websocket: WebSocket):
    """接收音频块，积累语音段，静音结束后触发 ASR → LLM → TTS 管道"""
    try:
        user_data = manager.user_data[client_id]
        current_time = time.time()

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
            if silence_duration >= SILENCE_THRESHOLD:
                user_data['is_speech_active'] = False
                buffered_audio = b''.join(user_data['speech_buffer'])
                user_data['speech_buffer'] = []
                logger.debug(f"客户端 {client_id} 语音段结束，积累 {len(buffered_audio)} 字节")
                if len(buffered_audio) >= MIN_SPEECH_BYTES:
                    if not user_data.get('is_processing', False):
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


async def tts_worker(queue: asyncio.Queue, client_id: str, websocket: WebSocket):
    """从队列中依次取句子生成并发送 TTS 音频"""
    while True:
        sentence = await queue.get()
        if sentence is None:
            break
        try:
            await handle_tts_request(sentence, client_id, websocket)
        except Exception as e:
            logger.error(f"TTS worker 错误 - 客户端 {client_id}: {e}")


async def process_speech(audio_data: bytes, client_id: str, websocket: WebSocket):
    """对完整语音段执行 ASR → LLM（流式）→ TTS（并行）管道"""
    if not voice_processor or not voice_processor.asr_model:
        logger.warning("ASR模型未初始化")
        return

    try:
        result = voice_processor.asr_model.generate(input=audio_data)
        if not (result and len(result) > 0 and 'text' in result[0]):
            return

        raw_text = result[0]['text'].strip()
        recognized_text = ASR_TAG_PATTERN.sub('', raw_text).strip()
        if not recognized_text:
            return

        logger.info(f"ASR识别 - 客户端 {client_id}: {recognized_text}")

        conversation_history = manager.user_data[client_id]['conversation_history']
        conversation_history.append({
            'type': 'user',
            'text': recognized_text,
            'timestamp': time.time()
        })

        if not (voice_processor and voice_processor.llm_client):
            await websocket.send_json({
                'type': 'llm_unavailable',
                'message': 'LLM服务暂不可用',
                'timestamp': time.time()
            })
            return

        await websocket.send_json({
            'type': 'llm_stream_start',
            'text': recognized_text,
            'role': 'user',
            'timestamp': time.time()
        })

        tts_queue: asyncio.Queue = asyncio.Queue()
        tts_task = asyncio.create_task(tts_worker(tts_queue, client_id, websocket))

        llm_response = ""
        sentence_buffer = ""
        _tts_choice = getattr(settings, 'TTS_MODEL', 'ChatTTS')
        if _tts_choice == 'EdgeTTS':
            tts_enabled = True
        elif _tts_choice == 'MiniMax':
            tts_enabled = bool(getattr(settings, 'MINIMAX_API_KEY', '').strip())
        else:
            tts_enabled = bool(voice_processor and voice_processor.tts_model)

        llm_stream_ok = False
        try:
            async for chunk in voice_processor.stream_llm_response(recognized_text, conversation_history):
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
            conversation_history.append({
                'type': 'assistant',
                'text': llm_response,
                'timestamp': time.time()
            })
            logger.info(f"LLM回复 - 客户端 {client_id}: {llm_response[:60]}...")

    except Exception as e:
        logger.error(f"语音处理管道错误 - 客户端 {client_id}: {e}")
        await websocket.send_json({
            'type': 'llm_error',
            'message': f'处理失败: {str(e)}'
        })