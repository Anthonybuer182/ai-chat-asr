import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """应用配置类"""
    
    # API配置
    API_KEY: str = ""  # 留空，从环境变量读取
    API_BASE: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    MODEL_NAME: str = "qwen-turbo"
    CURRENT_LIVE2D_MODEL: str = ""
    # 语音处理配置
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    AUDIO_FORMAT: str = "wav"
    # 实时识别：最后一帧语音后连续静音达到该秒数则整段送 ASR（略小响应更快，略大更不易切成两半）
    speech_end_silence_sec: float = 0.55
    
    # 唤醒词配置
    WAKEUP_KEYWORD: str = "你好小助手"
    WAKEUP_THRESHOLD: float = 0.8
    WAKEUP_TIMEOUT: int = 60

    # TTS模型配置
    TTS_MODEL: str = "EdgeTTS"
    # MiniMax 语音合成（HTTPS API，不占本地显卡）；文档: https://platform.minimax.io/docs/api-reference/speech-t2a-http
    MINIMAX_API_KEY: str = ""
    # 与 API Key 同页或账户信息中的 GroupId / 组织 ID；Postman 的 Headers 里若有 GroupId，须填此项
    MINIMAX_GROUP_ID: str = ""
    # 语音 API 网关：国内多为 api-bj.minimaxi.com，国际文档常见 api.minimax.io，须与控制台密钥所属区域一致
    MINIMAX_API_BASE: str = "https://api-bj.minimaxi.com"
    # 与官方示例一致时可设 speech-2.8-hd；旧版可用 speech-2.6-turbo
    MINIMAX_TTS_MODEL: str = "speech-2.8-hd"
    MINIMAX_VOICE_ID: str = "female-tianmei"
    # 留空则请求体中不传该字段（与部分 Postman 用例一致）；可填 Chinese / auto 等
    MINIMAX_LANGUAGE_BOOST: str = ""
    # speech-2.8 等模型支持，如 happy；留空则不传
    MINIMAX_VOICE_EMOTION: str = ""
    # speech-2.8-hd / speech-2.8-turbo 句内语气词：(laughs) 等；按 [EMOTION:] 自动注入（见 minimax_emotion_interjection）
    MINIMAX_EMOTION_INTERJECTION_ENABLED: bool = True
    MINIMAX_AUDIO_FORMAT: str = "mp3"  # mp3 / wav / flac（非流式）
    MINIMAX_AUDIO_SAMPLE_RATE: int = 32000

    
    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # 模型配置
    asr_model: str = "iic/SenseVoiceSmall"
    # 仅对 SenseVoice 等多语模型有效：zh 锁定中文（推荐本仓库中文界面）；多语场景可设 auto / en / ja / ko / yue
    asr_language: str = "zh"
    voiceprint_model: str = "damo/speech_campplus_sv_zh-cn_16k-common"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# 全局配置实例
settings = Settings()

def get_settings() -> Settings:
    """获取配置实例"""
    return settings

def update_settings(**kwargs):
    """更新配置"""
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)

def validate_config() -> bool:
    """验证配置是否有效"""
    try:
        # 检查必要的配置项
        if not settings.API_KEY:
            print("⚠️  警告: API_KEY未设置，LLM功能将不可用")
            print("💡 提示: 请设置环境变量 API_KEY 或创建 .env 文件")
            print("📝 示例: API_KEY=your_actual_api_key_here")
        
        if settings.SAMPLE_RATE not in [8000, 16000, 44100]:
            print(f"⚠️  警告: 采样率 {settings.SAMPLE_RATE} 不标准，建议使用8000、16000或44100")
            print("💡 提示: 将自动调整为16000")
            settings.SAMPLE_RATE = 16000
        
        if settings.CHANNELS not in [1, 2]:
            print(f"⚠️  警告: 声道数 {settings.CHANNELS} 无效，将使用单声道")
            settings.CHANNELS = 1
        
        # 检查模型配置
        if not settings.asr_model:
            print("⚠️  警告: ASR模型未配置，将使用默认模型")
        
        print("✅ 配置验证完成，系统可以启动")
        return True
    except Exception as e:
        print(f"❌ 配置验证失败: {e}")
        print("💡 提示: 系统将继续启动，但部分功能可能受限")
        return True  # 即使验证失败也允许启动