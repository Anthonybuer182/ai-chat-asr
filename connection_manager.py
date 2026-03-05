import asyncio
import logging
import time
from typing import Dict, Any, List
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_data: Dict[str, Dict[str, Any]] = {}
        self.interrupt_flags: Dict[str, bool] = {}  # 中断标志
        self.current_tts_tasks: Dict[str, asyncio.Task] = {}  # 当前TTS任务
        self.audio_processing_lock: Dict[str, asyncio.Lock] = {}  # 音频处理锁
        self._shutting_down = False  # 添加关闭标志

    async def connect(self, websocket: WebSocket, client_id: str):
        """建立WebSocket连接"""
        # 检查是否正在关闭
        if self._shutting_down:
            await websocket.close(code=1000, reason="服务器正在关闭")
            return
            
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.user_data[client_id] = {
            'is_listening': False,
            'is_speaking': False,
            'conversation_history': [],
            'last_activity': time.time(),
            'voiceprint': None,
            'wakeup_detected': False,
            'interrupt_enabled': True,  # 启用语音打断
            'keyword_wakeup_enabled': False,  # 默认关闭关键词唤醒
            'voiceprint_match_enabled': False,  # 默认关闭声纹匹配
            'vad_sensitivity': 0.3,  # VAD灵敏度
            'interrupt_threshold': 0.6,  # 打断阈值
            'conversation_state': 'idle'  # 对话状态: idle, listening, speaking, interrupted
        }
        self.interrupt_flags[client_id] = False
        self.audio_processing_lock[client_id] = asyncio.Lock()
        logger.info(f"客户端 {client_id} 已连接")

    def disconnect(self, client_id: str):
        """断开WebSocket连接"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.user_data:
            del self.user_data[client_id]
        logger.info(f"客户端 {client_id} 已断开")

    async def send_message(self, message: Dict[str, Any], client_id: str):
        """向指定客户端发送消息"""
        if client_id in self.active_connections and not self._shutting_down:
            try:
                await self.active_connections[client_id].send_json(message)
                self.user_data[client_id]['last_activity'] = time.time()
            except Exception as e:
                logger.error(f"发送消息到 {client_id} 失败: {e}")
                self.disconnect(client_id)

    def update_activity(self, client_id: str):
        """更新客户端活动时间"""
        if client_id in self.user_data:
            self.user_data[client_id]['last_activity'] = time.time()

    def get_inactive_clients(self, timeout: int = 300) -> List[str]:
        """获取超时未活动的客户端"""
        current_time = time.time()
        inactive_clients = []
        
        for client_id, data in self.user_data.items():
            if current_time - data['last_activity'] > timeout:
                inactive_clients.append(client_id)
        
        return inactive_clients

    async def cleanup_inactive_clients(self, timeout: int = 300):
        """清理超时未活动的客户端"""
        inactive_clients = self.get_inactive_clients(timeout)
        for client_id in inactive_clients:
            logger.info(f"清理超时客户端: {client_id}")
            self.disconnect(client_id)

    async def set_interrupt_flag(self, client_id: str, interrupt: bool = True):
        """设置中断标志"""
        if client_id in self.interrupt_flags:
            self.interrupt_flags[client_id] = interrupt
            if interrupt:
                self.user_data[client_id]['conversation_state'] = 'interrupted'
                logger.info(f"客户端 {client_id} 语音打断已触发")

    def get_interrupt_flag(self, client_id: str) -> bool:
        """获取中断标志"""
        return self.interrupt_flags.get(client_id, False)

    async def cancel_current_tts(self, client_id: str):
        """取消当前TTS任务"""
        if client_id in self.current_tts_tasks:
            task = self.current_tts_tasks[client_id]
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.info(f"客户端 {client_id} 的TTS任务已取消")
                except Exception as e:
                    logger.error(f"取消TTS任务时出错: {e}")
                finally:
                    del self.current_tts_tasks[client_id]

    async def set_conversation_state(self, client_id: str, state: str):
        """设置对话状态"""
        if client_id in self.user_data:
            self.user_data[client_id]['conversation_state'] = state
            logger.debug(f"客户端 {client_id} 对话状态更新为: {state}")

    def get_conversation_state(self, client_id: str) -> str:
        """获取对话状态"""
        return self.user_data.get(client_id, {}).get('conversation_state', 'idle')

    async def shutdown(self):
        """优雅关闭所有连接和任务"""
        self._shutting_down = True
        
        # 取消所有TTS任务
        for client_id in list(self.current_tts_tasks.keys()):
            await self.cancel_current_tts(client_id)
        
        # 关闭所有WebSocket连接
        for client_id, websocket in list(self.active_connections.items()):
            try:
                await websocket.close(code=1000, reason="服务器关闭")
            except Exception as e:
                logger.error(f"关闭WebSocket连接 {client_id} 时出错: {e}")
            finally:
                self.disconnect(client_id)
        
        logger.info("连接管理器已关闭")