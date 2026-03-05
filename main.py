import asyncio
import logging
import time
import os
import sys
import argparse
from typing import Dict, Any, List
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from funasr import AutoModel
from config import settings
import ChatTTS
import numpy as np
import torch

# 导入新的路由模块
from api_routes import api_router
from websocket_routes import init_websocket_routes
from connection_manager import ConnectionManager
from voice_processor import VoiceProcessor

# 设置模型缓存目录（在导入任何模型相关库之前）
os.environ['MODELSCOPE_CACHE'] = './modelscope_cache'  # 魔搭社区模型缓存
os.environ['HUGGINGFACE_HUB_CACHE'] = './huggingface_cache'  # Hugging Face模型缓存

# 确保缓存目录存在
for cache_dir in ['./modelscope_cache', './huggingface_cache']:
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 配置FastAPI应用，启用Swagger文档
app = FastAPI(
    title="实时语音对话系统",
    description="基于FastAPI的实时语音对话系统，支持语音识别、语音合成、声纹识别和Live2D模型管理",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI文档
    redoc_url="/redoc",  # ReDoc文档
    openapi_url="/openapi.json"  # OpenAPI规范文件
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# 创建连接管理器实例
manager = ConnectionManager()



# 创建语音处理器实例
voice_processor = VoiceProcessor()

# 初始化路由
# 注入全局变量到API路由
from api_routes import init_api_routes
init_api_routes(voice_processor, manager)

# 注册API路由
app.include_router(api_router)

# 初始化WebSocket路由
init_websocket_routes(app, voice_processor, manager)

# 添加静态文件服务
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/live2d/models", StaticFiles(directory="live2d_models"), name="live2d_models")

# 添加应用生命周期管理
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    logger.info("🚀 应用启动中...")
    # 异步初始化模型（不阻塞启动）
    asyncio.create_task(voice_processor.initialize_models())

@app.on_event("shutdown") 
async def shutdown_event():
    """应用关闭时清理资源"""
    logger.info("🛑 应用关闭中，清理资源...")
    # 优雅关闭连接管理器
    await manager.shutdown()
    logger.info("✅ 应用已关闭")

# 主页路由
@app.get("/", summary="主页面")
async def read_index():
    """主页面"""
    return FileResponse("static/index.html")

# 健康检查路由
@app.get("/health", summary="健康检查")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }

# 检查依赖是否安装
def check_dependencies():
    """检查依赖是否安装"""
    required_packages = [
        ('fastapi', 'FastAPI框架'),
        ('uvicorn', 'ASGI服务器'),
        ('websockets', 'WebSocket支持'),
        ('numpy', '数值计算'),
        ('pydantic', '数据验证'),
        ('pydantic_settings', '配置管理'),
        ('funasr', '语音识别'),
        ('torch', '深度学习框架')
        # 暂时跳过chattts检查，因为导入名称可能有问题
    ]
    
    missing_packages = []
    available_packages = []
    
    for package, description in required_packages:
        try:
            __import__(package.replace('-', '_'))
            available_packages.append((package, description))
        except ImportError:
            missing_packages.append((package, description))
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for package, description in missing_packages:
            print(f"   - {package}: {description}")
        print("\n💡 请使用以下命令安装:")
        print(f"pip install {' '.join([p[0] for p in missing_packages])}")
        return False
    else:
        print("✅ 所有依赖包已安装")
        return True

if __name__ == "__main__":
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="实时语音对话系统")
    parser.add_argument("--host", default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--reload", action="store_true", help="启用热重载")
    
    args = parser.parse_args()
    
    # 启动服务器
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )