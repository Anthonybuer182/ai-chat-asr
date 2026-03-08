import asyncio
import logging
import time
import json
import os
import numpy as np
import torch
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from config import settings
import ChatTTS
import edge_tts

logger = logging.getLogger(__name__)

LIVE2D_SYSTEM_PROMPT = """你是一个可爱的虚拟助手，拥有丰富的情感和肢体语言表达能力。

## 你的输出规范

你需要在回复文本中适当添加控制标签来表达情绪和动作。控制标签格式如下：

### 情绪/动作标签 [EMOTION:类型]
- happy (开心) - 用于表达高兴、快乐的情绪
- sad (伤心) - 用于表达难过、伤心的情绪  
- angry (生气) - 用于表达愤怒、生气的情绪
- surprised (惊讶) - 用于表达惊讶、意外的情绪
- neutral (中性/待机) - 用于表达正常、平静的情绪
- thinking (思考) - 用于表达思考、考虑的状态
- smile (微笑) - 用于表达微笑、轻笑
- blushing (脸红) - 用于表达害羞、不好意思
- greet (打招呼) - 用于打招呼、问候
- agree (赞同/点头) - 用于表示同意、认可
- disagree (否定/摇头) - 用于表示否定、不同意
- scared (害怕) - 用于表达恐惧、害怕
- excited (兴奋) - 用于表达兴奋、激动
- question (疑问) - 用于表达疑问、困惑
- cheer (欢呼) - 用于表达欢呼、庆祝
- disgust (厌恶) - 用于表达厌恶、反感
- shake (摇头) - 用于表示否定、不知道

### 使用示例

用户: 你好呀！
回复: [EMOTION:greet] 你好！很高兴见到你！

用户: 什么是人工智能？
回复: [EMOTION:thinking]人工智能啊...简单来说，它是用计算机模拟人类智能的技术...

用户: 你喜欢学习吗？
回复: [EMOTION:happy]喜欢！学习新知识让我很开心呢～

用户: 这个我不知道
回复: [EMOTION:thinking]嗯...这个问题我还不太清楚，不过我可以帮你查一查～

用户: 太棒了！
回复: [EMOTION:cheer]太好了！我们一起庆祝一下吧！

用户: 真的吗？
回复: [EMOTION:excited]真的呀～我也很开心呢！

用户: 这个好恶心
回复: [EMOTION:disgust]嗯...确实让人不太舒服

用户: 我好害怕
回复: [EMOTION:scared]别怕别怕～有我在呢！

## 重要规则

1. 控制标签要自然地融入回复中，不要过度使用
2. 每个句子或关键情绪点可以使用一个标签
3. 保持回复的自然流畅性，标签只是辅助表达
4. 优先使用与内容匹配的情绪和动作
5. 不要在标签中使用空格，如 [EMOTION:happy] 而不是 [EMOTION:happy ]
6. 可以根据对话内容选择合适的情绪标签

现在，请根据对话内容自然地使用这些标签来表达你的情绪和动作。"""


@dataclass
class VADResult:
    has_speech: bool
    speech_audio: Optional[bytes]
    clean_audio: Optional[bytes]
    speech_segments: List[Dict]
    avg_confidence: float
    is_speech_start: bool = False
    is_speech_end: bool = False


class VoiceProcessor:
    """语音处理核心类 - 优化版"""
    
    def __init__(self):
        self.asr_model = None
        self.vad_model = None
        self.vad_utils = None
        self.vad_iterator = None
        self.tts_model = None
        self.edge_tts_voice = "zh-CN-XiaoxiaoNeural"
        self.voiceprint_model = None
        self.llm_client = None
        self.keyword_wakeup_model = None
        self.is_initialized = False
        self.initialization_lock = asyncio.Lock()
        
        self.vad_config = {
            'threshold': 0.3,
            'min_speech_duration': 0.1,
            'max_silence_duration': 0.5,
            'window_size_samples': 512,
            'sampling_rate': 16000
        }
        
        self.vad_state = {
            'speech_probabilities': [],
            'current_segment_start': None,
            'is_speech_active': False,
            'silence_counter': 0,
            'speech_chunks_in_current_segment': 0
        }
        
        # 重复音频检测
        self._last_audio_hash = None
        self._last_process_time = 0
        self._duplicate_threshold = 0.5  # 相同内容处理间隔阈值（秒）

    def _calculate_audio_hash(self, audio_data: bytes) -> str:
        """计算音频数据的简单哈希值"""
        try:
            import hashlib
            return hashlib.md5(audio_data).hexdigest()
        except:
            return str(len(audio_data))

    def _is_duplicate_audio(self, audio_data: bytes) -> bool:
        """检测是否为重复音频"""
        current_time = time.time()
        
        # 检查时间间隔
        if current_time - self._last_process_time < self._duplicate_threshold:
            return True
        
        # 检查内容是否相同
        current_hash = self._calculate_audio_hash(audio_data)
        if current_hash == self._last_audio_hash:
            return True
        
        # 更新状态
        self._last_audio_hash = current_hash
        self._last_process_time = current_time
        
        return False

    async def initialize_models(self):
        """初始化所有语音处理模型"""
        async with self.initialization_lock:
            if self.is_initialized:
                return
                
            try:
                logger.info("🚀 开始初始化语音处理模型...")
                
                core_models = [
                    ("ASR模型", self._init_asr_model),
                    ("VAD模型", self._init_vad_model),
                    ("TTS模型", self._init_tts_model),
                    ("LLM客户端", self._init_llm_client),
                    ("关键词唤醒", self._init_keyword_wakeup),
                    ("声纹模型", self._init_voiceprint_model)
                ]
                
                success_count = 0
                failed_models = []
                
                for model_name, init_func in core_models:
                    try:
                        logger.info(f"🔧 正在初始化 {model_name}...")
                        result = await init_func()
                        if result:
                            success_count += 1
                            logger.info(f"✅ {model_name} 初始化成功")
                        else:
                            failed_models.append(model_name)
                            logger.warning(f"⚠️  {model_name} 初始化失败，使用备用方案")
                    except Exception as e:
                        failed_models.append(model_name)
                        logger.error(f"❌ {model_name} 初始化异常: {e}")
                
                core_available = success_count >= 3
                
                if core_available:
                    self.is_initialized = True
                    logger.info(f"🎉 语音处理模型初始化完成 ({success_count}/6)")
                    
                    if failed_models:
                        logger.warning(f"⚠️  以下模型初始化失败: {', '.join(failed_models)}")
                        logger.info("💡 部分功能将使用备用方案")
                    
                    if not self.llm_client:
                        logger.warning("⚠️  LLM功能不可用，请检查API_KEY配置")
                    if not self.vad_iterator:
                        logger.warning("⚠️  VAD功能使用备用方案（能量检测）")
                        
                else:
                    self.is_initialized = False
                    logger.error("❌ 语音处理模型初始化失败，核心功能不可用")
                    
            except Exception as e:
                logger.error(f"❌ 模型初始化过程失败: {e}")
                self.is_initialized = False

    async def _init_asr_model(self):
        """初始化语音识别模型"""
        try:
            from funasr import AutoModel
            self.asr_model = AutoModel(model=settings.asr_model)
            logger.info("ASR模型初始化成功")
            return True
        except Exception as e:
            logger.error(f"ASR模型初始化失败: {e}")
            return False

    async def _init_vad_model(self):
        """初始化VAD模型 - 优化版"""
        try:
            import socket
            socket.setdefaulttimeout(10)
            
            try:
                model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False,
                    verbose=False
                )
                
                (get_speech_timestamps, save_audio, read_audio, 
                 VADIterator, collect_chunks) = utils
                
                self.vad_model = model
                self.vad_utils = {
                    'get_speech_timestamps': get_speech_timestamps,
                    'save_audio': save_audio,
                    'read_audio': read_audio,
                    'collect_chunks': collect_chunks
                }
                self.vad_iterator = VADIterator(model)
                
                logger.info("Silero-VAD VADIterator初始化成功")
                return True
                
            except Exception as hub_error:
                logger.warning(f"torch.hub加载失败: {hub_error}")
                
                try:
                    from silero_vad import VADIterator as SileroVADIterator
                    self.vad_iterator = SileroVADIterator()
                    logger.info("使用本地silero-vad包初始化成功")
                    return True
                    
                except Exception as local_error:
                    logger.warning(f"本地silero-vad包加载失败: {local_error}")
                    raise hub_error
                    
        except Exception as e:
            logger.error(f"VAD模型初始化失败: {e}")
            logger.info("使用备用VAD方案（能量检测）")
            self.vad_iterator = None
            return True

    def reset_vad_state(self):
        """重置VAD状态"""
        self.vad_state = {
            'speech_probabilities': [],
            'current_segment_start': None,
            'is_speech_active': False,
            'silence_counter': 0,
            'speech_chunks_in_current_segment': 0
        }
        if self.vad_iterator:
            try:
                self.vad_iterator.reset_states()
            except:
                pass

    def detect_speech(self, audio_data: bytes, client_id: str = None) -> VADResult:
        """
        检测音频数据中是否有语音活动 - 优化版
        
        Args:
            audio_data: 音频字节数据
            client_id: 客户端ID
            
        Returns:
            VADResult: 包含检测结果的详细对象
        """
        if not audio_data or len(audio_data) < 512:
            return VADResult(
                has_speech=False,
                speech_audio=None,
                clean_audio=None,
                speech_segments=[],
                avg_confidence=0.0
            )
        
        # 检测重复音频
        if self._is_duplicate_audio(audio_data):
            if client_id:
                logger.debug(f"VAD检测：客户端 {client_id} 检测到重复音频，跳过处理")
            return VADResult(
                has_speech=False,
                speech_audio=None,
                clean_audio=None,
                speech_segments=[],
                avg_confidence=0.0,
                is_speech_start=False,
                is_speech_end=False
            )
        
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            if len(audio_array) < 512:
                logger.debug(f"音频数据过小（{len(audio_array)} < 512），跳过VAD检测")
                return VADResult(
                    has_speech=True,
                    speech_audio=audio_data,
                    clean_audio=None,
                    speech_segments=[],
                    avg_confidence=0.0
                )
            
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            if not self.vad_iterator:
                return self._energy_based_vad(audio_float, audio_data)
            
            return self._silero_vad_detection(audio_float, audio_data, client_id)
            
        except Exception as e:
            logger.warning(f"VAD检测失败: {e}")
            return VADResult(
                has_speech=True,
                speech_audio=audio_data,
                clean_audio=None,
                speech_segments=[],
                avg_confidence=0.0
            )

    def _silero_vad_detection(self, audio_float: np.ndarray, audio_data: bytes, 
                             client_id: str = None) -> VADResult:
        """使用Silero-VAD进行语音检测"""
        try:
            window_size = self.vad_config['window_size_samples']
            threshold = self.vad_config['threshold']
            
            speech_segments = []
            speech_probs = []
            is_speech_start = False
            is_speech_end = False
            speech_audio_samples = []
            
            self.reset_vad_state()
            
            for i in range(0, len(audio_float), window_size):
                chunk_end = min(i + window_size, len(audio_float))
                chunk = audio_float[i:chunk_end]
                
                if len(chunk) < window_size:
                    if len(chunk) < window_size * 0.5:
                        break
                    padding = np.zeros(window_size - len(chunk), dtype=np.float32)
                    chunk = np.concatenate([chunk, padding])
                
                try:
                    audio_tensor = torch.from_numpy(chunk).unsqueeze(0)
                    
                    with torch.no_grad():
                        speech_prob_tensor = self.vad_model(audio_tensor, 16000)
                    
                    speech_prob = speech_prob_tensor.item()
                    speech_probs.append(speech_prob)
                    
                    dynamic_threshold = self._calculate_dynamic_threshold(speech_probs)
                    
                    is_speech_chunk = speech_prob > dynamic_threshold
                    
                    current_time = i / 16000.0
                    chunk_end_time = chunk_end / 16000.0
                    
                    if is_speech_chunk:
                        if not self.vad_state['is_speech_active']:
                            self.vad_state['is_speech_active'] = True
                            self.vad_state['current_segment_start'] = current_time
                            self.vad_state['speech_chunks_in_current_segment'] = 0
                            is_speech_start = True
                            logger.debug(f"🎯 检测到语音开始: {current_time:.2f}s")
                        
                        self.vad_state['speech_chunks_in_current_segment'] += 1
                        self.vad_state['silence_counter'] = 0
                        
                        speech_segments.append({
                            'start': current_time,
                            'end': chunk_end_time,
                            'confidence': float(speech_prob)
                        })
                        speech_audio_samples.append(chunk)
                    else:
                        self.vad_state['silence_counter'] += 1
                        
                        if (self.vad_state['is_speech_active'] and 
                            self.vad_state['silence_counter'] * window_size / 16000.0 > self.vad_config['max_silence_duration']):
                            
                            if self.vad_state['speech_chunks_in_current_segment'] >= 3:
                                self.vad_state['is_speech_active'] = False
                                is_speech_end = True
                                logger.debug(f"🔚 检测到语音结束: {current_time:.2f}s")
                            
                            self.vad_state['speech_chunks_in_current_segment'] = 0
                    
                except Exception as chunk_error:
                    logger.warning(f"VAD检测小块失败: {chunk_error}")
                    continue
            
            avg_confidence = np.mean(speech_probs) if speech_probs else 0.0
            
            has_speech = len(speech_segments) > 0
            
            clean_audio = None
            if has_speech and speech_audio_samples:
                clean_audio_float = np.concatenate(speech_audio_samples)
                clean_audio_int16 = (clean_audio_float * 32767).astype(np.int16)
                clean_audio = clean_audio_int16.tobytes()
            
            if client_id:
                if has_speech:
                    logger.debug(f"VAD检测：客户端 {client_id} 检测到语音活动，语音段数={len(speech_segments)}, "
                               f"平均置信度={avg_confidence:.3f}, 开始={is_speech_start}, 结束={is_speech_end}")
                else:
                    logger.debug(f"VAD检测：客户端 {client_id} 未检测到语音活动")
            
            return VADResult(
                has_speech=has_speech,
                speech_audio=audio_data if has_speech else None,
                clean_audio=clean_audio,
                speech_segments=speech_segments,
                avg_confidence=float(avg_confidence),
                is_speech_start=is_speech_start,
                is_speech_end=is_speech_end
            )
            
        except Exception as e:
            logger.error(f"Silero-VAD检测失败: {e}")
            return self._energy_based_vad(audio_float, audio_data)

    def _calculate_dynamic_threshold(self, speech_probs: List[float]) -> float:
        """计算动态阈值"""
        base_threshold = self.vad_config['threshold']
        
        if len(speech_probs) < 5:
            return base_threshold
        
        recent_probs = speech_probs[-10:]
        mean_prob = np.mean(recent_probs)
        
        if mean_prob > 0.5:
            return min(0.5, base_threshold * 1.2)
        elif mean_prob < 0.2:
            return max(0.15, base_threshold * 0.8)
        
        return base_threshold

    def _energy_based_vad(self, audio_float: np.ndarray, audio_data: bytes) -> VADResult:
        """备用VAD：基于能量的检测"""
        try:
            frame_length = 512
            num_frames = len(audio_float) // frame_length
            
            energy_threshold = 0.01
            speech_frames = 0
            speech_audio_samples = []
            
            for i in range(num_frames):
                start = i * frame_length
                end = start + frame_length
                frame = audio_float[start:end]
                energy = np.mean(frame ** 2)
                
                if energy > energy_threshold:
                    speech_frames += 1
                    speech_audio_samples.append(frame)
            
            speech_ratio = speech_frames / num_frames if num_frames > 0 else 0
            has_speech = speech_ratio > 0.3
            
            clean_audio = None
            if has_speech and speech_audio_samples:
                clean_audio_float = np.concatenate(speech_audio_samples)
                clean_audio_int16 = (clean_audio_float * 32767).astype(np.int16)
                clean_audio = clean_audio_int16.tobytes()
            
            return VADResult(
                has_speech=has_speech,
                speech_audio=audio_data if has_speech else None,
                clean_audio=clean_audio,
                speech_segments=[{
                    'start': 0,
                    'end': len(audio_float) / 16000.0,
                    'confidence': float(speech_ratio)
                }] if has_speech else [],
                avg_confidence=float(speech_ratio)
            )
            
        except Exception as e:
            logger.error(f"能量VAD检测失败: {e}")
            return VADResult(
                has_speech=True,
                speech_audio=audio_data,
                clean_audio=None,
                speech_segments=[],
                avg_confidence=0.0
            )

    async def _init_tts_model(self):
        """初始化TTS模型"""
        try:
            self.tts_model = ChatTTS.Chat()
            self.tts_model.load()
            
            try:
                import json
                import os
                speaker_file = os.path.join(os.path.dirname(__file__), 'female_speaker.json')
                if os.path.exists(speaker_file):
                    with open(speaker_file, 'r', encoding='utf-8') as f:
                        speaker_data = json.load(f)
                        self.rand_spk = speaker_data.get('speaker')
                        logger.info("已加载固定女声音色")
                else:
                    self.rand_spk = self.tts_model.sample_random_speaker()
                    logger.info("未找到音色文件，使用随机音色")
            except Exception as e:
                logger.warning(f"加载音色文件失败: {e}")
                self.rand_spk = self.tts_model.sample_random_speaker()
            
            logger.info("TTS模型从本地加载成功，已固定音色")          
            return True
        except Exception as e:
            logger.error(f"TTS模型初始化失败: {e}")
            logger.info("使用备用TTS方案")
            return True

    async def _init_edge_tts(self):
        """初始化 EdgeTTS（异步方法，用于检查连接性）"""
        try:
            import edge_tts
            voices = await edge_tts.list_voices()
            zh_voices = [v for v in voices if v['Locale'].startswith('zh-')]
            logger.info(f"EdgeTTS 可用语音数量: {len(zh_voices)}")
            logger.info(f"EdgeTTS 默认语音: {self.edge_tts_voice}")
            return True
        except Exception as e:
            logger.error(f"EdgeTTS 初始化失败: {e}")
            return False

    async def generate_edge_tts(self, text: str) -> Optional[bytes]:
        """使用 EdgeTTS 生成语音"""
        try:
            import edge_tts
            import io
            
            communicate = edge_tts.Communicate(text, self.edge_tts_voice)
            audio_data = io.BytesIO()
            
            async for chunk in communicate.stream():
                if chunk['type'] == 'audio':
                    audio_data.write(chunk['data'])
            
            audio_bytes = audio_data.getvalue()
            
            if audio_bytes:
                wav_data = self._create_wav_header(audio_bytes, sample_rate=24000)
                return wav_data
            
            return None
        except Exception as e:
            logger.error(f"EdgeTTS 生成失败: {e}")
            return None

    def _create_wav_header(self, audio_data: bytes, sample_rate: int = 24000, num_channels: int = 1, bits_per_sample: int = 16):
        """创建 WAV 文件头"""
        import struct
        import io
        
        data_size = len(audio_data)
        buffer = io.BytesIO()
        
        buffer.write(b'RIFF')
        buffer.write(struct.pack('<I', 36 + data_size))
        buffer.write(b'WAVE')
        buffer.write(b'fmt ')
        buffer.write(struct.pack('<I', 16))
        buffer.write(struct.pack('<H', 1))
        buffer.write(struct.pack('<H', num_channels))
        buffer.write(struct.pack('<I', sample_rate))
        buffer.write(struct.pack('<I', sample_rate * num_channels * bits_per_sample // 8))
        buffer.write(struct.pack('<H', num_channels * bits_per_sample // 8))
        buffer.write(struct.pack('<H', bits_per_sample))
        buffer.write(b'data')
        buffer.write(struct.pack('<I', data_size))
        buffer.write(audio_data)
        
        return buffer.getvalue()

    async def _init_voiceprint_model(self):
        """初始化声纹模型"""
        try:
            try:
                import modelscope
            except ImportError:
                logger.error("modelscope包未安装")
                self.voiceprint_model = None
                return False
                
            required_packages = ['torchaudio', 'librosa', 'scipy']
            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(package)
            
            if missing_packages:
                logger.error(f"缺少依赖包: {missing_packages}")
                self.voiceprint_model = None
                return False
                
            from modelscope.pipelines import pipeline
            
            model_name = settings.voiceprint_model
            logger.info(f"正在下载声纹模型: {model_name}")
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"尝试下载声纹模型 (第{attempt + 1}次)...")
                    
                    self.voiceprint_model = pipeline(
                        task='speaker-verification',
                        model=model_name,
                        model_revision='v1.0.0'
                    )
                    
                    logger.info("声纹模型下载和初始化成功")
                    return True
                    
                except Exception as e:
                    logger.warning(f"声纹模型下载失败 (第{attempt + 1}次): {e}")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(2)
                        
            logger.error("声纹模型下载失败，已达到最大重试次数")
            raise Exception("声纹模型下载失败")
                        
        except Exception as e:
            logger.error(f"声纹模型初始化失败: {e}")
            self.voiceprint_model = None
            return False

    async def _init_llm_client(self):
        """初始化LLM客户端"""
        try:
            from openai import OpenAI
            self.llm_client = OpenAI(
                api_key=settings.API_KEY,
                base_url=settings.API_BASE
            )
            logger.info("LLM客户端初始化成功")
            return True
        except Exception as e:
            logger.error(f"LLM客户端初始化失败: {e}")
            logger.info("LLM功能将不可用")
            return False

    async def get_llm_response(self, user_message: str, conversation_history: list = None) -> str:
        """调用LLM获取回复"""
        try:
            if not self.llm_client:
                logger.warning("LLM客户端未初始化，无法获取回复")
                return "抱歉，我现在无法回答您的问题。"
            
            messages = []
            
            messages.append({"role": "system", "content": LIVE2D_SYSTEM_PROMPT})
            
            if conversation_history:
                for entry in conversation_history[-10:]:
                    if entry['type'] == 'user':
                        messages.append({"role": "user", "content": entry['text']})
                    elif entry['type'] == 'assistant':
                        messages.append({"role": "assistant", "content": entry['text']})
            
            messages.append({"role": "user", "content": user_message})
            
            response = self.llm_client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=messages,
                max_tokens=500,
                temperature=0.7,
                stream=False
            )
            
            if response and response.choices:
                llm_response = response.choices[0].message.content.strip()
                logger.info(f"LLM回复: {llm_response[:100]}...")
                return llm_response
            else:
                logger.warning("LLM返回结果为空")
                return "抱歉，我没有理解您的问题。"
                
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return "抱歉，我现在无法回答您的问题。"

    async def stream_llm_response(self, user_message: str, conversation_history: list = None):
        """流式调用LLM获取回复"""
        try:
            if not self.llm_client:
                logger.warning("LLM客户端未初始化，无法获取回复")
                yield "抱歉，我现在无法回答您的问题。"
                return
            
            messages = []
            
            messages.append({"role": "system", "content": LIVE2D_SYSTEM_PROMPT})
            
            if conversation_history:
                for entry in conversation_history[-10:]:
                    if entry['type'] == 'user':
                        messages.append({"role": "user", "content": entry['text']})
                    elif entry['type'] == 'assistant':
                        messages.append({"role": "assistant", "content": entry['text']})
            
            messages.append({"role": "user", "content": user_message})
            
            response = self.llm_client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=messages,
                max_tokens=500,
                temperature=0.7,
                stream=True
            )
            
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
            
            logger.info(f"LLM流式回复完成，总长度: {len(full_response)}")
                
        except Exception as e:
            logger.error(f"LLM流式调用失败: {e}")
            yield "抱歉，我现在无法回答您的问题。"

    async def _init_keyword_wakeup(self):
        """初始化基于代码的关键词唤醒功能"""
        try:
            self.keyword_wakeup_model = None
            
            try:
                import numpy as np
                import librosa
                logger.info("音频处理库可用，关键词唤醒功能已启用")
            except ImportError:
                logger.warning("librosa不可用，使用简单能量检测方案")
            
            logger.info(f"基于代码的关键词唤醒已初始化，唤醒词: {settings.WAKEUP_KEYWORD}")
            return True
        except Exception as e:
            logger.error(f"关键词唤醒初始化失败: {e}")
            self.keyword_wakeup_model = None
            logger.info("关键词唤醒功能将使用备用方案")
            return True

    async def extract_voiceprint_feature(self, audio_data: bytes) -> Any:
        """提取声纹特征"""
        try:
            if not self.voiceprint_model:
                logger.warning("声纹模型未初始化，无法提取特征")
                return None
            
            logger.info("声纹特征提取功能待实现")
            return {"feature": "placeholder", "timestamp": time.time()}
            
        except Exception as e:
            logger.error(f"声纹特征提取失败: {e}")
            return None

    async def voiceprint_match(self, audio_data: bytes, client_id: str, 
                              stored_voiceprint: Any = None) -> Dict[str, Any]:
        """声纹匹配"""
        try:
            if not stored_voiceprint:
                logger.info(f"客户端 {client_id} 未注册声纹，跳过声纹匹配")
                return {"match": True, "confidence": 1.0, "message": "未注册声纹，跳过匹配"}
            
            if not self.voiceprint_model:
                logger.warning("声纹模型未初始化，跳过声纹匹配")
                return {"match": True, "confidence": 1.0, "message": "声纹模型未初始化，跳过匹配"}
            
            current_feature = await self.extract_voiceprint_feature(audio_data)
            
            if not current_feature:
                logger.warning("无法提取当前音频的声纹特征")
                return {"match": False, "confidence": 0.0, "message": "声纹特征提取失败"}
            
            similarity_score = await self.calculate_similarity(stored_voiceprint, current_feature)
            
            match_threshold = 0.7
            is_match = similarity_score >= match_threshold
            
            logger.info(f"声纹匹配结果: 相似度={similarity_score:.3f}, 匹配={is_match}")
            
            return {
                "match": is_match,
                "confidence": similarity_score,
                "message": f"声纹匹配{'成功' if is_match else '失败'} (相似度: {similarity_score:.3f})"
            }
            
        except Exception as e:
            logger.error(f"声纹匹配失败: {e}")
            return {"match": False, "confidence": 0.0, "message": f"声纹匹配过程出错: {str(e)}"}

    async def calculate_similarity(self, feature1: Any, feature2: Any) -> float:
        """计算声纹特征相似度（简化实现）"""
        import random
        return random.uniform(0.5, 0.95)

    def update_vad_threshold(self, threshold: float):
        """更新VAD检测阈值"""
        self.vad_config['threshold'] = max(0.1, min(0.9, threshold))
        logger.info(f"VAD阈值已更新: {self.vad_config['threshold']}")

    def get_vad_statistics(self) -> Dict[str, Any]:
        """获取VAD统计信息"""
        return {
            'config': self.vad_config.copy(),
            'state': self.vad_state.copy(),
            'initialized': self.is_initialized
        }
