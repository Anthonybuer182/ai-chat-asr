import os
import time
import json
import zipfile
import shutil
from typing import List, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, Request, HTTPException, Body
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import logging

from config import update_settings, settings as app_settings

logger = logging.getLogger(__name__)

# 创建API路由器
api_router = APIRouter(prefix="/api", tags=["API接口"])

# 全局变量（将在main.py中注入）
voice_processor = None
manager = None

def init_api_routes(voice_proc, conn_manager):
    """初始化API路由，注入全局变量"""
    global voice_processor, manager
    voice_processor = voice_proc
    manager = conn_manager

# 声纹存储目录
VOICEPRINT_DIR = "voiceprints"
LIVE2D_MODELS_DIR = "live2d_models"

# 确保目录存在
for directory in [VOICEPRINT_DIR, LIVE2D_MODELS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

# 声纹相关API
@api_router.post("/voiceprint/upload", summary="上传声纹音频文件")
async def upload_voiceprint(voiceprint: UploadFile = File(...), client_id: str = Form(...)):
    """上传声纹音频文件"""
    try:
        # 验证文件类型
        if not voiceprint.filename.endswith(('.webm', '.wav', '.mp3', '.ogg')):
            return {
                "success": False,
                "message": "不支持的文件格式，请上传音频文件"
            }
        
        # 使用固定文件名，每次覆盖之前的文件
        filename = "current_voiceprint.webm"
        filepath = os.path.join(VOICEPRINT_DIR, filename)
        
        # 读取并保存文件
        contents = await voiceprint.read()
        with open(filepath, "wb") as f:
            f.write(contents)
        
        logger.info(f"声纹文件已保存: {filepath}, 大小: {len(contents)} 字节")
        
        # 提取声纹特征（如果声纹模型可用）
        if voice_processor and voice_processor.voiceprint_model:
            try:
                voiceprint_feature = await voice_processor.extract_voiceprint_feature(contents)
                
                if voiceprint_feature:
                    # 保存声纹特征到用户数据
                    manager.user_data[client_id]['voiceprint'] = voiceprint_feature
                    logger.info(f"声纹特征已提取并保存，客户端: {client_id}")
                
            except Exception as e:
                logger.warning(f"声纹特征提取失败: {e}")
        
        return {
            "success": True,
            "message": "声纹上传成功",
            "filename": filename,
            "size": len(contents)
        }
        
    except Exception as e:
        logger.error(f"声纹上传失败: {e}")
        return {
            "success": False,
            "message": f"声纹上传失败: {str(e)}"
        }

@api_router.get("/voiceprint/status", summary="获取声纹文件状态")
async def get_voiceprint_status():
    """获取声纹文件状态"""
    try:
        # 检查声纹目录中是否有文件
        voiceprint_files = [f for f in os.listdir(VOICEPRINT_DIR) if f.endswith(('.webm', '.wav', '.mp3', '.ogg'))]
        
        if voiceprint_files:
            # 获取最新的声纹文件信息
            latest_file = max(voiceprint_files, key=lambda f: os.path.getmtime(os.path.join(VOICEPRINT_DIR, f)))
            filepath = os.path.join(VOICEPRINT_DIR, latest_file)
            
            file_info = {
                "filename": latest_file,
                "size": os.path.getsize(filepath),
                "uploadTime": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(filepath))),
                "filepath": filepath
            }
            
            return {
                "success": True,
                "hasFile": True,
                "fileInfo": file_info,
                "message": "已上传声纹文件"
            }
        else:
            return {
                "success": True,
                "hasFile": False,
                "fileInfo": None,
                "message": "未上传声纹文件"
            }
            
    except Exception as e:
        logger.error(f"获取声纹文件状态失败: {e}")
        return {
            "success": False,
            "hasFile": False,
            "fileInfo": None,
            "message": f"获取声纹文件状态失败: {str(e)}"
        }

@api_router.post("/voiceprint/delete", summary="删除声纹文件")
async def delete_voiceprint(request: Request):
    """删除声纹文件"""
    try:
        filename = None
        # 处理JSON请求体
        try:
            request_data = await request.json()
            filename = request_data.get('filename')
        except:
            # 如果没有JSON数据，则filename保持为None（删除所有文件）
            pass
        
        deleted_files = []
        
        if filename:
            # 删除指定文件
            filepath = os.path.join(VOICEPRINT_DIR, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
                deleted_files.append(filename)
                logger.info(f"声纹文件已删除: {filename}")
        else:
            # 删除所有文件
            for f in os.listdir(VOICEPRINT_DIR):
                if f.endswith(('.webm', '.wav', '.mp3', '.ogg')):
                    filepath = os.path.join(VOICEPRINT_DIR, f)
                    os.remove(filepath)
                    deleted_files.append(f)
                    logger.info(f"声纹文件已删除: {f}")
        
        # 检查是否删除了当前使用的声纹文件
        is_current_file = False
        if filename:
            # 这里需要检查该文件是否是当前使用的声纹文件
            # 简化处理：假设如果删除的是最新文件，则认为是当前文件
            voiceprint_files = [f for f in os.listdir(VOICEPRINT_DIR) if f.endswith(('.webm', '.wav', '.mp3', '.ogg'))]
            if not voiceprint_files or filename not in voiceprint_files:
                is_current_file = True
        
        if deleted_files:
            return {
                "success": True,
                "message": f"成功删除 {len(deleted_files)} 个声纹文件",
                "deletedFiles": deleted_files,
                "is_current_file": is_current_file
            }
        else:
            return {
                "success": True,
                "message": "没有找到可删除的声纹文件",
                "deletedFiles": [],
                "is_current_file": False
            }
            
    except Exception as e:
        logger.error(f"删除声纹文件失败: {e}")
        return {
            "success": False,
            "message": f"删除声纹文件失败: {str(e)}",
            "deletedFiles": [],
            "is_current_file": False
        }

@api_router.get("/voiceprint/files", summary="获取所有声纹文件列表")
async def get_voiceprint_files():
    """获取所有声纹文件列表"""
    try:
        logger.info(f"开始获取声纹文件列表，目录: {VOICEPRINT_DIR}")
        voiceprint_files = []
        
        # 检查目录是否存在
        if not os.path.exists(VOICEPRINT_DIR):
            logger.warning(f"声纹目录不存在: {VOICEPRINT_DIR}")
            return {
                "success": True,
                "files": [],
                "count": 0,
                "message": "声纹目录不存在"
            }
        
        # 获取目录中的文件
        files_in_dir = os.listdir(VOICEPRINT_DIR)
        logger.info(f"目录中的文件: {files_in_dir}")
        
        for filename in files_in_dir:
            if filename.endswith(('.webm', '.wav', '.mp3', '.ogg')):
                filepath = os.path.join(VOICEPRINT_DIR, filename)
                if os.path.isfile(filepath):
                    file_info = {
                        "filename": filename,
                        "size": os.path.getsize(filepath),
                        "uploadTime": int(os.path.getmtime(filepath) * 1000),  # 转换为毫秒
                        "filepath": filepath
                    }
                    voiceprint_files.append(file_info)
                    logger.info(f"添加声纹文件: {filename}, 大小: {file_info['size']} 字节")
        
        # 按上传时间倒序排列
        voiceprint_files.sort(key=lambda x: x["uploadTime"], reverse=True)
        
        logger.info(f"成功获取到 {len(voiceprint_files)} 个声纹文件")
        
        return {
            "success": True,
            "files": voiceprint_files,
            "count": len(voiceprint_files),
            "message": f"找到 {len(voiceprint_files)} 个声纹文件"
        }
        
    except Exception as e:
        logger.error(f"获取声纹文件列表失败: {e}")
        return {
            "success": False,
            "files": [],
            "count": 0,
            "message": f"获取声纹文件列表失败: {str(e)}"
        }

@api_router.get("/voiceprint/play", summary="播放声纹文件")
async def play_voiceprint(filename: str):
    """播放声纹文件"""
    try:
        # 验证文件名安全性
        if '..' in filename or '/' in filename or '\\' in filename:
            return {
                "success": False,
                "message": "无效的文件名"
            }
        
        filepath = os.path.join(VOICEPRINT_DIR, filename)
        
        if not os.path.exists(filepath):
            return {
                "success": False,
                "message": "文件不存在"
            }
        
        # 返回音频文件
        return FileResponse(
            filepath,
            media_type="audio/webm",
            filename=filename
        )
        
    except Exception as e:
        logger.error(f"播放声纹文件失败: {e}")
        return {
            "success": False,
            "message": f"播放声纹文件失败: {str(e)}"
        }

# Live2D模型相关API
@api_router.post("/live2d/upload", summary="上传Live2D模型压缩包")
async def upload_live2d_model(zip_file: UploadFile = File(...)):
    """上传Live2D模型压缩包"""
    try:
        # 验证文件类型
        if not zip_file.filename.endswith(('.zip', '.rar', '.7z')):
            return {
                "success": False,
                "message": "不支持的文件格式，请上传.zip、.rar或.7z格式的压缩包"
            }
        
        # 创建Live2D模型目录
        os.makedirs(LIVE2D_MODELS_DIR, exist_ok=True)
        
        # 读取压缩包内容
        contents = await zip_file.read()
        
        # 保存临时压缩包文件
        temp_zip_path = os.path.join(LIVE2D_MODELS_DIR, "temp_upload.zip")
        with open(temp_zip_path, "wb") as f:
            f.write(contents)
        
        # 解压压缩包
        # 获取模型名称（使用压缩包文件名，去掉扩展名）
        model_name = os.path.splitext(zip_file.filename)[0]
        
        # 先检查压缩包内的文件结构
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            # 获取压缩包内所有文件的路径
            file_list = zip_ref.namelist()
            
            # 检查压缩包内是否已经包含模型名称的目录
            has_model_dir = any(f.startswith(model_name + '/') for f in file_list)
            
            if has_model_dir:
                # 如果压缩包内已经包含模型目录，直接解压到live2d_models目录
                model_dir = LIVE2D_MODELS_DIR
                # 如果目录已存在，先删除
                if os.path.exists(os.path.join(model_dir, model_name)):
                    shutil.rmtree(os.path.join(model_dir, model_name))
            else:
                # 如果压缩包内没有模型目录，创建模型目录
                model_dir = os.path.join(LIVE2D_MODELS_DIR, model_name)
                # 如果目录已存在，先删除
                if os.path.exists(model_dir):
                    shutil.rmtree(model_dir)
                os.makedirs(model_dir, exist_ok=True)
            
            # 解压文件
            zip_ref.extractall(model_dir)
        
        # 删除临时压缩包
        os.remove(temp_zip_path)
        
        # 验证解压后的文件结构
        model_files = []
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if file.endswith(('.model3.json', '.json', '.moc', '.mtn', '.physics.json', '.exp.json')):
                    model_files.append(os.path.join(root, file))
        
        if not model_files:
            shutil.rmtree(model_dir)
            return {
                "success": False,
                "message": "压缩包中未找到有效的Live2D模型文件"
            }
        
        logger.info(f"Live2D模型上传成功: {model_name}, 包含 {len(model_files)} 个模型文件")
        
        return {
            "success": True,
            "message": f"模型 {model_name} 上传成功",
            "modelName": model_name,
            "fileCount": len(model_files)
        }
        
    except zipfile.BadZipFile:
        return {
            "success": False,
            "message": "压缩包文件损坏或格式不正确"
        }
    except Exception as e:
        logger.error(f"上传Live2D模型失败: {e}")
        return {
            "success": False,
            "message": f"上传Live2D模型失败: {str(e)}"
        }

@api_router.get("/live2d/models", summary="获取Live2D模型列表")
async def get_live2d_models():
    """获取Live2D模型列表"""
    try:
        if not os.path.exists(LIVE2D_MODELS_DIR):
            return {
                "success": True,
                "models": [],
                "message": "Live2D模型目录不存在"
            }
        
        models = []
        
        # 遍历live2d_models目录
        for item in os.listdir(LIVE2D_MODELS_DIR):
            item_path = os.path.join(LIVE2D_MODELS_DIR, item)
            if os.path.isdir(item_path):
                # 检查是否是Live2D模型目录（包含.model3.json文件）
                model_files = []
                for root, dirs, files in os.walk(item_path):
                    for file in files:
                        if file.endswith('.model3.json'):
                            model_files.append(os.path.join(root, file))
                
                if model_files:
                    # 使用第一个找到的model3.json文件
                    model_file = model_files[0]
                    model_info = {
                        "name": item,
                        "fileName": os.path.basename(model_file),
                        "filePath": model_file,
                        "fileType": "json",
                        "uploadTime": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(model_file)))
                    }
                    models.append(model_info)
        
        return {
            "success": True,
            "models": models,
            "message": f"找到 {len(models)} 个Live2D模型"
        }
        
    except Exception as e:
        logger.error(f"获取Live2D模型列表失败: {e}")
        return {
            "success": False,
            "models": [],
            "message": f"获取Live2D模型列表失败: {str(e)}"
        }

@api_router.get("/live2d/model-config/{model_name}", summary="获取Live2D模型动画配置")
async def get_live2d_model_config(model_name: str):
    """获取Live2D模型的动画配置"""
    try:
        model_dict_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_dict.json")
        
        if not os.path.exists(model_dict_path):
            return {
                "success": False,
                "message": "模型配置文件不存在"
            }
        
        with open(model_dict_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        models_config = data.get("models", {})
        model_config = models_config.get(model_name)
        
        if not model_config:
            return {
                "success": False,
                "message": f"未找到模型 {model_name} 的配置"
            }
        
        return {
            "success": True,
            "model_name": model_name,
            "config": model_config
        }
    except Exception as e:
        logger.error(f"获取Live2D模型配置失败: {e}")
        return {
            "success": False,
            "message": f"获取模型配置失败: {str(e)}"
        }

@api_router.get("/live2d/all-config", summary="获取所有Live2D模型动画配置")
async def get_all_live2d_model_config():
    """获取所有Live2D模型的动画配置"""
    try:
        model_dict_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_dict.json")
        
        if not os.path.exists(model_dict_path):
            return {
                "success": False,
                "message": "模型配置文件不存在"
            }
        
        with open(model_dict_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return {
            "success": True,
            "models": data.get("models", {}),
            "default_model": data.get("default_model", "Epsilon")
        }
    except Exception as e:
        logger.error(f"获取Live2D模型配置失败: {e}")
        return {
            "success": False,
            "message": f"获取模型配置失败: {str(e)}"
        }

@api_router.delete("/live2d/models/{model_name}", summary="删除Live2D模型")
async def delete_live2d_model(model_name: str):
    """删除Live2D模型"""
    try:
        # 安全验证：确保模型名称不包含路径遍历字符
        if '..' in model_name or '/' in model_name or '\\' in model_name:
            return {
                "success": False,
                "message": "无效的模型名称"
            }
        
        model_dir = os.path.join(LIVE2D_MODELS_DIR, model_name)
        
        # 检查目录是否存在
        if not os.path.exists(model_dir):
            logger.warning(f"尝试删除不存在的模型目录: {model_dir}")
            return {
                "success": False,
                "message": f"模型目录不存在: {model_name}"
            }
        
        # 安全验证：确保路径在live2d_models目录内
        abs_model_dir = os.path.abspath(model_dir)
        abs_live2d_dir = os.path.abspath(LIVE2D_MODELS_DIR)
        
        if not abs_model_dir.startswith(abs_live2d_dir):
            logger.error(f"路径安全验证失败: {abs_model_dir} 不在 {abs_live2d_dir} 内")
            return {
                "success": False,
                "message": "无效的模型路径"
            }
        
        # 删除模型目录
        shutil.rmtree(model_dir)
        
        logger.info(f"Live2D模型已删除: {model_name}")
        
        return {
            "success": True,
            "message": f"模型 {model_name} 已删除"
        }
        
    except Exception as e:
        logger.error(f"删除Live2D模型失败: {e}")
        return {
            "success": False,
            "message": f"删除Live2D模型失败: {str(e)}"
        }

# 系统状态API
@api_router.get("/status", summary="获取系统状态")
async def get_system_status():
    """获取系统状态信息"""
    try:
        status_info = {
            "system": {
                "timestamp": time.time(),
                "uptime": time.time() - os.path.getctime(__file__) if os.path.exists(__file__) else 0
            },
            "models": {
                "asr_initialized": voice_processor.asr_model is not None if voice_processor else False,
                "vad_initialized": voice_processor.vad_iterator is not None if voice_processor else False,
                "tts_initialized": voice_processor.tts_model is not None if voice_processor else False,
                "voiceprint_initialized": voice_processor.voiceprint_model is not None if voice_processor else False
            },
            "storage": {
                "voiceprint_files": len([f for f in os.listdir(VOICEPRINT_DIR) if f.endswith(('.webm', '.wav', '.mp3', '.ogg'))]) if os.path.exists(VOICEPRINT_DIR) else 0,
                "live2d_models": len([d for d in os.listdir(LIVE2D_MODELS_DIR) if os.path.isdir(os.path.join(LIVE2D_MODELS_DIR, d))]) if os.path.exists(LIVE2D_MODELS_DIR) else 0
            },
            "connections": {
                "active_clients": len(manager.active_connections) if manager else 0
            }
        }
        
        return {
            "success": True,
            "data": status_info,
            "message": "系统状态获取成功"
        }
        
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        return {
            "success": False,
            "data": {},
            "message": f"获取系统状态失败: {str(e)}"
        }

@api_router.get("/settings/get", summary="获取系统设置")
async def get_settings():
    """获取系统设置"""
    try:
        return {
            "success": True,
            "data": {
                "TTS_MODEL": app_settings.TTS_MODEL,
                "WAKEUP_KEYWORD": app_settings.WAKEUP_KEYWORD,
                "WAKEUP_THRESHOLD": app_settings.WAKEUP_THRESHOLD,
                "MODEL_NAME": app_settings.MODEL_NAME,
                "CURRENT_LIVE2D_MODEL": app_settings.CURRENT_LIVE2D_MODEL
            },
            "message": "获取系统设置成功"
        }
    except Exception as e:
        logger.error(f"获取系统设置失败: {e}")
        return {
            "success": False,
            "data": {},
            "message": f"获取系统设置失败: {str(e)}"
        }

@api_router.post("/settings/change", summary="修改系统设置")
async def change_settings(settings_dict: Dict[str, Any] = Body(...)):
    """修改系统设置"""
    try:
        valid_keys = ["TTS_MODEL", "WAKEUP_KEYWORD", "WAKEUP_THRESHOLD", "MODEL_NAME", "CURRENT_LIVE2D_MODEL"]
        valid_tts_models = ["ChatTTS", "EdgeTTS"]
        
        update_data = {}
        errors = []
        
        for key, value in settings_dict.items():
            if key not in valid_keys:
                errors.append(f"无效的配置项: {key}")
                continue
            
            if key == "TTS_MODEL" and value not in valid_tts_models:
                errors.append(f"无效的TTS模型: {value}，可选值: {valid_tts_models}")
                continue
            
            update_data[key] = value
        
        if errors:
            return {
                "success": False,
                "message": "; ".join(errors)
            }
        
        update_settings(**update_data)
        logger.info(f"系统设置已更新: {update_data}")
        
        return {
            "success": True,
            "data": {
                "TTS_MODEL": app_settings.TTS_MODEL,
                "WAKEUP_KEYWORD": app_settings.WAKEUP_KEYWORD,
                "WAKEUP_THRESHOLD": app_settings.WAKEUP_THRESHOLD,
                "MODEL_NAME": app_settings.MODEL_NAME,
                "CURRENT_LIVE2D_MODEL": app_settings.CURRENT_LIVE2D_MODEL
            },
            "message": "系统设置更新成功"
        }
    except Exception as e:
        logger.error(f"修改系统设置失败: {e}")
        return {
            "success": False,
            "message": f"修改系统设置失败: {str(e)}"
        }