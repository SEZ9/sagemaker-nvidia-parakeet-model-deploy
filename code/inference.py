"""
NVIDIA Parakeet TDT 0.6B v2 - SageMaker 推理代码
"""

import os
import json
import base64
import tempfile
import logging
import time
from typing import Any, Dict, List, Union

import torch
import nemo.collections.asr as nemo_asr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局模型
model = None
device = None


def model_fn(model_dir: str):
    """加载模型"""
    global model, device

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading model on device: {device}")

    # 从 HuggingFace 加载预训练模型
    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v2"
    )
    model.to(device)
    model.eval()

    # 预热
    logger.info("Warming up model...")
    warmup_audio = torch.randn(16000).numpy()  # 1秒静音
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        import soundfile as sf
        sf.write(f.name, warmup_audio, 16000)
        try:
            model.transcribe([f.name])
        except:
            pass
        os.unlink(f.name)

    logger.info("Model loaded successfully")
    return model


def input_fn(request_body: str, request_content_type: str) -> Dict[str, Any]:
    """解析输入"""
    if request_content_type == "application/json":
        return json.loads(request_body)
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data: Dict[str, Any], model) -> Dict[str, Any]:
    """推理"""
    start_time = time.time()
    temp_file = None

    try:
        # 获取音频数据
        audio_source = input_data.get("audio")
        language = input_data.get("language", "en")
        timestamps = input_data.get("timestamps", False)

        if not audio_source:
            return {"error": "Missing 'audio' field"}

        # 处理不同的音频输入格式
        audio_data = None

        if isinstance(audio_source, str):
            if audio_source.startswith("data:"):
                # Data URL 格式
                base64_data = audio_source.split(",", 1)[1] if "," in audio_source else audio_source
                audio_data = base64.b64decode(base64_data)
            elif "/" in audio_source or "\\" in audio_source:
                # 文件路径（仅限容器内部）
                with open(audio_source, "rb") as f:
                    audio_data = f.read()
            else:
                # 纯 Base64
                audio_data = base64.b64decode(audio_source)

        elif isinstance(audio_source, dict):
            if "data" in audio_source:
                base64_data = audio_source["data"]
                if base64_data.startswith("data:"):
                    base64_data = base64_data.split(",", 1)[1]
                audio_data = base64.b64decode(base64_data)
            elif "path" in audio_source:
                # URL 或路径
                path = audio_source["path"]
                if path.startswith(("http://", "https://")):
                    import requests
                    resp = requests.get(path, timeout=30)
                    resp.raise_for_status()
                    audio_data = resp.content
                else:
                    with open(path, "rb") as f:
                        audio_data = f.read()

        if audio_data is None:
            return {"error": "Failed to load audio"}

        # 写入临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            temp_file = f.name

        # 转录
        if timestamps:
            output = model.transcribe([temp_file], timestamps=True)
            result = {
                "text": output[0].text,
                "timestamps": output[0].timestamp
            }
        else:
            output = model.transcribe([temp_file])
            result = {
                "text": output[0].text if hasattr(output[0], 'text') else output[0]
            }

        processing_time = time.time() - start_time
        result["processing_time"] = round(processing_time, 3)
        result["model"] = "nvidia/parakeet-tdt-0.6b-v2"

        return result

    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        return {"error": str(e)}

    finally:
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)


def output_fn(prediction: Dict[str, Any], response_content_type: str) -> str:
    """格式化输出"""
    return json.dumps(prediction, ensure_ascii=False)
