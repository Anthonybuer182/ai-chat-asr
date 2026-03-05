# 实时语音对话系统

这是一个基于FastAPI WebSocket的实时语音对话系统，支持语音识别、语音唤醒、声纹匹配、双工对话等功能。

## 功能特性

- 🎤 实时音频流处理（bytes格式传输）
- 🔔 语音关键词唤醒
- 👤 声纹匹配识别
- 💬 双工对话和语音打断
- 📊 VAD语音活动检测
- 🗣️ ASR语音转文本
- 🤖 LLM智能回复生成
- 🔊 TTS文本转语音

## 技术栈

- **后端**: FastAPI + WebSocket
- **前端**: HTML5 + JavaScript + Web Audio API
- **语音识别**: fun-asr
- **声纹匹配**: Pipeline
- **语音检测**: silerovad
- **TTS**: ChatTTS

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行项目

1. 启动后端服务器：
```bash
python main.py
```

2. 打开浏览器访问：`http://localhost:8000`

## 配置说明

在`.env`文件中配置相关API密钥和参数：

```
API_KEY=your_api_key_here
API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_NAME=qwen-turbo
```