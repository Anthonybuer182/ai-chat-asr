import asyncio
import logging
import time
from typing import Dict, Any
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_data: Dict[str, Dict[str, Any]] = {}
        self.interrupt_flags: Dict[str, bool] = {}
        self.current_tts_tasks: Dict[str, asyncio.Task] = {}
        self._shutting_down = False

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
            'wakeup_state': 'always_on',
            'last_wakeup_time': 0.0,
            'interrupt_enabled': True,
            'vad_sensitivity': 0.3,
            'interrupt_threshold': 0.6,
            'conversation_state': 'idle',
            'speech_buffer': [],
            'last_speech_time': 0.0,
            'is_speech_active': False,
            'is_processing': False,
        }
        self.interrupt_flags[client_id] = False
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

    def get_wakeup_state(self, client_id: str) -> str:
        """获取唤醒状态：always_on / sleep / awake"""
        return self.user_data.get(client_id, {}).get('wakeup_state', 'always_on')

    def set_wakeup_state(self, client_id: str, state: str):
        """设置唤醒状态"""
        if client_id in self.user_data:
            self.user_data[client_id]['wakeup_state'] = state
            if state == 'awake':
                self.user_data[client_id]['last_wakeup_time'] = time.time()
            logger.info(f"客户端 {client_id} 唤醒状态切换为: {state}")

    def update_wakeup_activity(self, client_id: str):
        """更新唤醒活跃时间（每次对话完成时调用）"""
        if client_id in self.user_data:
            self.user_data[client_id]['last_wakeup_time'] = time.time()

    def check_wakeup_timeout(self, client_id: str, timeout_seconds: int = 60) -> bool:
        """检查是否唤醒超时，返回 True 表示已超时应进入休眠"""
        if client_id not in self.user_data:
            return True
        state = self.user_data[client_id].get('wakeup_state', 'always_on')
        if state != 'awake':
            return False
        last_time = self.user_data[client_id].get('last_wakeup_time', 0.0)
        if last_time == 0.0:
            return False
        return (time.time() - last_time) > timeout_seconds

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