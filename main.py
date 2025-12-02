"""
AI语音聊天系统后端

功能概述：
- 支持文本和语音两种聊天模式
- 集成AI进行智能对话
- 提供WebSocket实时通信

主要模块：
1. 配置管理 (Config)
2. 连接管理 (ConnectionManager)
3. 聊天历史管理 (ChatHistory)
4. AI服务 (AIService)
5. WebSocket路由处理

依赖安装：
pip install fastapi uvicorn websockets openai gtts faster-whisper numpy scipy python-multipart python-dotenv

作者：AI助手
版本：1.0.0
创建时间：2024年
"""

import os
import asyncio
import time
from typing import Dict, List, AsyncGenerator
from dataclasses import dataclass
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
import logging
from dotenv import load_dotenv


# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 性能监控装饰器
def log_performance(func):
    """
    性能监控装饰器，记录函数执行时间
    
    Args:
        func: 被装饰的函数
        
    Returns:
        wrapper: 包装后的函数
    """
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        # logger.info(f"开始执行函数: {func.__name__}")
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            # logger.info(f"函数 {func.__name__} 执行完成，耗时: {execution_time:.3f}秒")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            # logger.error(f"函数 {func.__name__} 执行失败，耗时: {execution_time:.3f}秒，错误: {str(e)}")
            raise
    
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        # logger.info(f"开始执行函数: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            # logger.info(f"函数 {func.__name__} 执行完成，耗时: {execution_time:.3f}秒")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            # logger.error(f"函数 {func.__name__} 执行失败，耗时: {execution_time:.3f}秒，错误: {str(e)}")
            raise
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

# 加载环境变量
load_dotenv()

# 配置管理类
@dataclass
class Config:
    """
    系统配置管理类
    
    属性说明：
    - API_KEY: API密钥，从环境变量读取
    - MODEL: 使用的模型名称
    - API_BASE: API基础URL
    - WHISPER_MODEL: Whisper语音识别模型大小 (tiny, base, small, medium, large)
    - SAMPLE_RATE: 音频采样率
    - SYSTEM_PROMPTS: 系统提示词模板，按语言和模式分类
    """
    
    API_KEY = os.getenv("API_KEY", "your-api-key-here")
    MODEL = os.getenv("MODEL", "deepseek-chat")
    API_BASE = os.getenv("API_BASE", "https://api.deepseek.com/v1")
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny, base, small, medium, large中文
    SAMPLE_RATE = 16000

    # 系统提示词模板
    SYSTEM_PROMPTS = {
        "zh": {
            "chat": "你叫小兰，是一个18岁的女大学生，性格活泼开朗，说话俏皮简洁，用中文简洁回答，限50字内，注意要纯文本输去除除式格式和表情。"
        },
        "en": {
            "chat": "I'm Xiao Lan, an 18-year-old university student. I'm bubbly and playful—keeping my answers short and sweet in English, under 50 words, plain text only, no formatting or emojis."
        }
    }
# 创建全局配置实例
config = Config()
logger.info("配置类初始化完成")

# 初始化OpenAI客户端（用于 API）
client = AsyncOpenAI(
    api_key=config.API_KEY,
    base_url=config.API_BASE
)

# 创建FastAPI应用
app = FastAPI(title="AI Voice Chat System")

# 配置静态文件服务
app.mount("/static", StaticFiles(directory="static"), name="static")

# 连接管理器
class ConnectionManager:
    """
    WebSocket连接管理器
    
    功能：
    - 管理不同频道的WebSocket连接
    - 跟踪活跃连接状态
    - 提供连接和断开连接的方法
    - 发送JSON消息到指定连接
    
    属性：
    - active_connections: 按频道分类的活跃连接字典
    """
    
    def __init__(self):
        """初始化连接管理器"""
        self.active_connections: Dict[str, List[WebSocket]] = {
            "chat": [],
            "voice": []
        }
        logger.info("连接管理器初始化完成")
    
    @log_performance
    async def connect(self, websocket: WebSocket, channel: str):
        """
        接受WebSocket连接并添加到活跃连接列表
        
        Args:
            websocket: WebSocket连接对象
            channel: 连接频道 ("chat" 或 "voice")
        """
        await websocket.accept()
        self.active_connections[channel].append(websocket)
        logger.info(f"客户端连接到 {channel} 频道，当前连接数: {len(self.active_connections[channel])}")
    
    def disconnect(self, websocket: WebSocket, channel: str):
        """
        从活跃连接列表中移除WebSocket连接
        
        Args:
            websocket: WebSocket连接对象
            channel: 连接频道 ("chat" 或 "voice")
        """
        if websocket in self.active_connections[channel]:
            self.active_connections[channel].remove(websocket)
            logger.info(f"客户端从 {channel} 频道断开连接，剩余连接数: {len(self.active_connections[channel])}")
        else:
            logger.warning(f"尝试断开不存在的连接: {channel} 频道")

# 创建全局连接管理器实例
manager = ConnectionManager()

# 聊天历史管理
class ChatHistory:
    """
    聊天历史管理器
    
    功能：
    - 按会话ID存储聊天历史
    - 限制历史记录长度
    - 提供历史记录的增删改查
    
    属性：
    - histories: 按会话ID分组的聊天历史字典
    - max_history: 每个会话的最大历史记录数
    """
    
    def __init__(self, max_history: int = 50):
        """
        初始化聊天历史管理器
        
        Args:
            max_history: 每个会话的最大历史记录数
        """
        self.histories: Dict[str, List[dict]] = {}
        self.max_history = max_history
        logger.info(f"聊天历史管理器初始化完成，最大历史记录数: {max_history}")
    
    @log_performance
    def add_message(self, session_id: str, role: str, content: str):
        """
        添加消息到指定会话的历史记录
        
        Args:
            session_id: 会话ID
            role: 消息角色 ("user" 或 "assistant")
            content: 消息内容
        """
        if session_id not in self.histories:
            self.histories[session_id] = []
        
        # 添加消息
        message = {
            "role": role,
            "content": content
        }
        self.histories[session_id].append(message)
        
        # 限制历史长度
        if len(self.histories[session_id]) > self.max_history * 2:
            old_length = len(self.histories[session_id])
            self.histories[session_id] = self.histories[session_id][-self.max_history:]
            new_length = len(self.histories[session_id])
            logger.debug(f"会话 {session_id} 历史记录从 {old_length} 条裁剪到 {new_length} 条")
        
        logger.debug(f"添加到会话 {session_id} 的消息: {role} - {content[:50]}...")
    
    @log_performance
    def get_history(self, session_id: str) -> List[dict]:
        """
        获取指定会话的历史记录
        
        Args:
            session_id: 会话ID
            
        Returns:
            List[dict]: 会话历史记录列表
        """
        history = self.histories.get(session_id, [])
        return history
    
    @log_performance
    def clear_history(self, session_id: str):
        """
        清除指定会话的历史记录
        
        Args:
            session_id: 会话ID
        """
        if session_id in self.histories:
            del self.histories[session_id]
            logger.info(f"已清除会话 {session_id} 的历史记录")
        else:
            logger.warning(f"尝试清除不存在的会话历史: {session_id}")

# 创建全局聊天历史管理器实例
chat_history = ChatHistory()


# AI服务
class AIService:
    """
    AI服务类，集成多种AI功能
    
    功能：
    - 聊天API调用
    - 流式聊天响应
    - 多种TTS引擎支持
    - Whisper语音识别
    - 按句子处理TTS
    """
    
    def __init__(self):
        """初始化AI服务"""
        logger.info("AI服务初始化完成")
    
    @staticmethod
    @log_performance
    async def get_chat_response(messages: List[dict]) -> str:
        """
        获取聊天API的非流式响应
        
        Args:
            messages: 聊天消息列表，包含角色和内容
            
        Returns:
            str: AI生成的响应内容
        """
        try:
            logger.info(f"开始调用API，消息数量: {len(messages)}")
            
            response = await client.chat.completions.create(
                model=config.MODEL,
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            logger.info(f"API调用成功，响应长度: {len(content)} 字符")
            return content
            
        except Exception as e:
            logger.error(f"API调用失败: {str(e)}")
            return "抱歉，我遇到了一些问题。请稍后再试。"
    
    @staticmethod
    @log_performance
    async def get_chat_response_stream(messages: List[dict]) -> AsyncGenerator[str, None]:
        """
        获取聊天API的流式响应
        
        Args:
            messages: 聊天消息列表，包含角色和内容
            
        Yields:
            str: 流式响应的文本块
        """
        try:
            response = await client.chat.completions.create(
                model=config.MODEL,
                messages=messages,
                stream=True,
                max_tokens=500,
                temperature=0.7
            )
            
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield content
        
        except Exception as e:
            logger.error(f"流式API调用失败: {str(e)}")
            yield "抱歉，我遇到了一些问题。请稍后再试。"
    
    

# 响应处理器类
class ResponseHandler:
    """
    响应处理器 - 封装文本和语音发送的重复逻辑
    
    功能：
    - 统一处理AI响应的文本流式发送
    - 统一处理TTS音频的异步生成和发送
    - 提供简洁的接口供WebSocket端点调用
    """
    
    def __init__(self, manager: ConnectionManager, ai_service: AIService):
        """
        初始化响应处理器
        
        Args:
            manager: WebSocket连接管理器
            audio_processor: 音频处理器实例
            ai_service: AI服务实例
        """
        self.manager = manager
        self.ai_service = ai_service
        logger.info("响应处理器初始化完成")
        

# 创建AIService实例
ai_service = AIService()

# 创建ResponseHandler实例
response_handler = ResponseHandler(manager, ai_service)

# 路由
@app.get("/")
async def get_index():
    """返回HTML页面"""
    return FileResponse("index.html")

# WebSocket端点 - 文本和语音聊天
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    聊天WebSocket端点 - 处理文本和音频消息的AI聊天功能
    
    支持的消息类型:
    - text: 文本消息，直接获取AI响应
    - audio: 音频消息，先进行语音识别再获取AI响应
    """
    session_id = f"chat_{id(websocket)}"
    logger.info(f"新的聊天WebSocket连接建立，客户端: {websocket.client}")
    await manager.connect(websocket, "chat")
    
    try:
        while True:
            # 接收消息
            logger.debug("等待接收WebSocket消息...")
            data = await websocket.receive_json()
            message_type = data.get("type")
            language = data.get("language", "zh")
            

    except WebSocketDisconnect:
        logger.info(f"聊天WebSocket连接断开，会话ID: {session_id}")
        manager.disconnect(websocket, "chat")
        chat_history.clear_history(session_id)
        logger.info(f"聊天会话 {session_id} 已断开连接并清空历史")
    except Exception as e:
        logger.error(f"聊天WebSocket错误: {str(e)}")
        manager.disconnect(websocket, "chat")

# 主函数
if __name__ == "__main__":
    """
    应用程序主入口点
    """
    import uvicorn
    
    logger.info("开始启动AI语音聊天服务器...")
    
    # 确保HTML文件存在
    html_file_path = "index.html"
    if not os.path.exists(html_file_path):
        logger.warning(f"HTML文件未找到: {html_file_path}，请确保已创建HTML文件")
    else:
        logger.info("HTML文件检查通过")
    
    # 启动服务器
    logger.info("正在启动uvicorn服务器...")
    logger.info("服务器配置 - 主机: localhost, 端口: 8000, 日志级别: info")
    
    try:
        uvicorn.run(
            app,
            host="localhost",
            port=8000,
            log_level="info"
        )
        logger.info("服务器启动成功，正在监听连接...")
    except Exception as e:
        logger.error(f"服务器启动失败: {str(e)}")
        raise