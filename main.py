import asyncio
import logging
import time
import os
import sys
import argparse
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api_routes import api_router, init_api_routes
from websocket_routes import init_websocket_routes
from connection_manager import ConnectionManager
from voice_processor import VoiceProcessor

os.environ['MODELSCOPE_CACHE'] = './modelscope_cache'

if not os.path.exists('./modelscope_cache'):
    os.makedirs('./modelscope_cache', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

manager = ConnectionManager()
voice_processor = VoiceProcessor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 应用启动中...")
    asyncio.create_task(voice_processor.initialize_models())
    yield
    logger.info("🛑 应用关闭中，清理资源...")
    await manager.shutdown()
    logger.info("✅ 应用已关闭")


app = FastAPI(
    title="实时语音对话系统",
    description="基于FastAPI的实时语音对话系统，支持语音识别、语音合成、声纹识别和Live2D模型管理",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_api_routes(voice_processor, manager)
app.include_router(api_router)
init_websocket_routes(app, voice_processor, manager)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/live2d/models", StaticFiles(directory="live2d_models"), name="live2d_models")


@app.get("/", summary="主页面")
async def read_index():
    return FileResponse("static/index.html")


@app.get("/health", summary="健康检查")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }


def check_dependencies():
    required_packages = [
        ('fastapi', 'FastAPI框架'),
        ('uvicorn', 'ASGI服务器'),
        ('websockets', 'WebSocket支持'),
        ('numpy', '数值计算'),
        ('pydantic', '数据验证'),
        ('pydantic_settings', '配置管理'),
        ('funasr', '语音识别'),
        ('torch', '深度学习框架')
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
    if not check_dependencies():
        sys.exit(1)

    parser = argparse.ArgumentParser(description="实时语音对话系统")
    parser.add_argument("--host", default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--reload", action="store_true", help="启用热重载")

    args = parser.parse_args()

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
