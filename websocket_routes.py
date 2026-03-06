import json
import logging
import time
import asyncio
from typing import Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
import base64
import ChatTTS
from voice_processor import VoiceProcessor
from connection_manager import ConnectionManager
from config import settings

logger = logging.getLogger(__name__)

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
    
    elif command == 'enable_keyword_wakeup':
        manager.user_data[client_id]['keyword_wakeup_enabled'] = True
        await websocket.send_json({
            'type': 'keyword_wakeup_enabled',
            'message': '关键词唤醒功能已启用',
            'timestamp': time.time()
        })
    
    elif command == 'disable_keyword_wakeup':
        manager.user_data[client_id]['keyword_wakeup_enabled'] = False
        await websocket.send_json({
            'type': 'keyword_wakeup_disabled',
            'message': '关键词唤醒功能已禁用',
            'timestamp': time.time()
        })
    
    elif command == 'enable_voiceprint_match':
        manager.user_data[client_id]['voiceprint_match_enabled'] = True
        await websocket.send_json({
            'type': 'voiceprint_match_enabled',
            'message': '声纹匹配功能已启用',
            'timestamp': time.time()
        })
    
    elif command == 'disable_voiceprint_match':
        manager.user_data[client_id]['voiceprint_match_enabled'] = False
        await websocket.send_json({
            'type': 'voiceprint_match_disabled',
            'message': '声纹匹配功能已禁用',
            'timestamp': time.time()
        })
    
    elif command == 'delete_message':
        message_index = text_message.get('index')
        message_type = text_message.get('type')
        
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
        tts_model = getattr(settings, 'TTS_MODEL', 'ChatTTS')
        
        if tts_model == 'EdgeTTS':
            await handle_edge_tts_request(sentence, websocket)
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
    """处理音频数据"""
    try:        
        print(f"处理音频")
        
        # 使用VoiceProcessor的detect_speech方法进行VAD语音活动检测
        if voice_processor:
            vad_result = voice_processor.detect_speech(audio_data, client_id)
            
            # 如果没有检测到语音活动，跳过ASR处理
            if not vad_result.has_speech:
                logger.debug(f"VAD检测：客户端 {client_id} 未检测到语音活动，跳过ASR")
                manager.update_activity(client_id)
                return
            
            speech_audio = vad_result.clean_audio if vad_result.clean_audio else vad_result.speech_audio
        else:
            speech_audio = audio_data
        
        # 处理音频数据（语音识别）
        try:
            if not voice_processor or not voice_processor.asr_model:
                logger.warning("ASR模型未初始化")
                return
                
            # 使用ASR模型进行语音识别（使用VAD检测到的有效语音数据）
            result = voice_processor.asr_model.generate(input=speech_audio)
            
            if result and len(result) > 0 and 'text' in result[0]:
                recognized_text = result[0]['text'].strip()
                
                if recognized_text:
                    print(f"ASR识别结果 - 客户端 {client_id}: {recognized_text}")
                    
                    conversation_history = manager.user_data[client_id]['conversation_history']
                    conversation_history.append({
                        'type': 'user',
                        'text': recognized_text,
                        'timestamp': time.time()
                    })
                    
                    if voice_processor and voice_processor.llm_client:
                        try:
                            await websocket.send_json({
                                'text': recognized_text,
                                'role': 'user',
                                'type': 'llm_stream_start',
                                'timestamp': time.time()
                            })
                            
                            llm_response = ""
                            sentence_buffer = ""
                            
                            async def process_sentence_tts(sentence: str):
                                """按顺序处理TTS任务"""
                                try:
                                    await handle_tts_request(sentence, client_id, websocket)
                                except Exception as tts_error:
                                    logger.error(f"TTS处理错误 - 客户端 {client_id}: {tts_error}")
                            
                            import re
                            sentence_endings = r'[。！？!?…]+'
                            
                            async for chunk in voice_processor.stream_llm_response(
                                recognized_text, 
                                conversation_history
                            ):
                                llm_response += chunk
                                sentence_buffer += chunk
                                
                                while re.search(sentence_endings, sentence_buffer):
                                    match = re.search(sentence_endings, sentence_buffer)
                                    sentence_end_pos = match.end()
                                    complete_sentence = sentence_buffer[:sentence_end_pos].strip()
                                    
                                    tts_model = getattr(settings, 'TTS_MODEL', 'ChatTTS')
                                    if complete_sentence and voice_processor and (voice_processor.tts_model or tts_model == 'EdgeTTS'):
                                        print(f"创建TTS任务: '{complete_sentence}'")
                                        await process_sentence_tts(complete_sentence)
                                    else:
                                        print(f"未创建TTS任务: complete_sentence={complete_sentence}, tts_model={voice_processor.tts_model if voice_processor else None}")
                                    
                                    sentence_buffer = sentence_buffer[sentence_end_pos:]
                            
                            await websocket.send_json({
                                'role': 'assistant',
                                'type': 'llm_stream_end',
                                'timestamp': time.time()
                            })
                            
                            if sentence_buffer.strip():
                                remaining_text = sentence_buffer.strip()
                                tts_model = getattr(settings, 'TTS_MODEL', 'ChatTTS')
                                if voice_processor and (voice_processor.tts_model or tts_model == 'EdgeTTS'):
                                    await process_sentence_tts(remaining_text)
                            
                            print(f"LLM回复 - 客户端 {client_id}: {llm_response[:50]}...")
                            
                            conversation_history.append({
                                'type': 'assistant',
                                'text': llm_response,
                                'timestamp': time.time()
                            })
                                
                        except Exception as llm_error:
                            logger.error(f"LLM调用错误 - 客户端 {client_id}: {llm_error}")
                            await websocket.send_json({
                                'type': 'llm_error',
                                'message': f'LLM调用失败: {str(llm_error)}'
                            })
                    else:
                        logger.warning(f"LLM不可用 - 客户端 {client_id}")
                        await websocket.send_json({
                            'type': 'llm_unavailable',
                            'message': 'LLM服务暂不可用',
                            'timestamp': time.time()
                        })
                        if voice_processor and voice_processor.tts_model:
                            fallback_response = "抱歉，我现在无法回答您的问题。"
                            await handle_tts_request(fallback_response, client_id, websocket)
                            
                else:
                    logger.debug(f"ASR识别结果为空 - 客户端 {client_id}")
            else:
                logger.debug(f"ASR处理结果无效 - 客户端 {client_id}")
        
        except Exception as e:
            logger.error(f"ASR处理错误 - 客户端 {client_id}: {e}")
    
    except Exception as e:
        logger.error(f"处理音频数据错误 - 客户端 {client_id}: {e}")
        logger.error(f"处理音频数据错误 - 客户端 {client_id}: {e}")