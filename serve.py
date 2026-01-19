#!/usr/bin/env python3
"""
Parakeet TDT 0.6B v2 - SageMaker 推理服务
"""

import os
import json
import base64
import tempfile
import logging
import time
from typing import Any, Dict, Optional

import torch
from flask import Flask, request, jsonify, Response
import nemo.collections.asr as nemo_asr

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全局模型
model: Optional[Any] = None
model_loaded = False


def load_model():
    """加载 Parakeet 模型"""
    global model, model_loaded

    if model_loaded:
        return model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading Parakeet TDT 0.6B v2 on device: {device}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    start_time = time.time()

    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v2"
    )
    model.to(device)
    model.eval()

    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f}s")

    # 预热
    logger.info("Warming up model...")
    try:
        import soundfile as sf
        import numpy as np
        warmup_audio = np.random.randn(16000).astype(np.float32) * 0.01
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            sf.write(f.name, warmup_audio, 16000)
            model.transcribe([f.name])
        logger.info("Warmup complete")
    except Exception as e:
        logger.warning(f"Warmup failed (non-critical): {e}")

    model_loaded = True
    return model


def process_audio(audio_source) -> Optional[bytes]:
    """处理各种音频输入格式"""
    audio_data = None

    try:
        if isinstance(audio_source, str):
            # Data URL 格式: data:audio/wav;base64,...
            if audio_source.startswith("data:"):
                base64_data = audio_source.split(",", 1)[1] if "," in audio_source else audio_source
                audio_data = base64.b64decode(base64_data)
            # 文件路径
            elif "/" in audio_source or "\\" in audio_source:
                with open(audio_source, "rb") as f:
                    audio_data = f.read()
            # 纯 Base64
            else:
                audio_data = base64.b64decode(audio_source)

        elif isinstance(audio_source, dict):
            if "data" in audio_source:
                base64_data = audio_source["data"]
                if base64_data.startswith("data:"):
                    base64_data = base64_data.split(",", 1)[1]
                audio_data = base64.b64decode(base64_data)
            elif "path" in audio_source:
                path = audio_source["path"]
                if path.startswith(("http://", "https://")):
                    import requests
                    resp = requests.get(path, timeout=60)
                    resp.raise_for_status()
                    audio_data = resp.content
                else:
                    with open(path, "rb") as f:
                        audio_data = f.read()

    except Exception as e:
        logger.error(f"Failed to process audio: {e}")
        return None

    return audio_data


@app.route("/ping", methods=["GET"])
def health_check():
    """SageMaker 健康检查"""
    if model_loaded:
        return Response(status=200)
    else:
        return Response(status=503)


@app.route("/invocations", methods=["POST"])
def invoke():
    """SageMaker 推理接口"""
    start_time = time.time()
    temp_file = None

    try:
        # 解析请求
        if request.content_type == "application/json":
            input_data = request.get_json()
        else:
            return jsonify({"error": f"Unsupported content type: {request.content_type}"}), 415

        # 获取参数
        audio_source = input_data.get("audio")
        timestamps = input_data.get("timestamps", False)
        language = input_data.get("language", "en")

        if not audio_source:
            return jsonify({"error": "Missing 'audio' field"}), 400

        # 处理音频
        audio_data = process_audio(audio_source)
        if audio_data is None:
            return jsonify({"error": "Failed to process audio data"}), 400

        # 写入临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            temp_file = f.name

        # 转录
        logger.info(f"Transcribing audio file: {len(audio_data)} bytes")

        if timestamps:
            output = model.transcribe([temp_file], timestamps=True)
            result = {
                "text": output[0].text,
                "timestamps": {
                    "word": [
                        {"word": w.word, "start": w.start, "end": w.end}
                        for w in output[0].timestamp.get("word", [])
                    ] if hasattr(output[0], 'timestamp') and output[0].timestamp else []
                }
            }
        else:
            output = model.transcribe([temp_file])
            text = output[0].text if hasattr(output[0], 'text') else str(output[0])
            result = {"text": text}

        processing_time = time.time() - start_time
        result["processing_time"] = round(processing_time, 3)
        result["model"] = "nvidia/parakeet-tdt-0.6b-v2"

        logger.info(f"Transcription completed in {processing_time:.3f}s")

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except:
                pass


@app.route("/", methods=["GET"])
def root():
    """根路径信息"""
    return jsonify({
        "model": "nvidia/parakeet-tdt-0.6b-v2",
        "status": "healthy" if model_loaded else "loading",
        "endpoints": {
            "/ping": "Health check",
            "/invocations": "Transcription endpoint"
        }
    })


if __name__ == "__main__":
    # 启动时加载模型
    logger.info("Starting Parakeet ASR server...")
    load_model()

    # 启动 Flask 服务
    port = int(os.environ.get("SAGEMAKER_BIND_TO_PORT", 8080))
    logger.info(f"Starting server on port {port}")

    # 生产环境用 gunicorn
    if os.environ.get("USE_GUNICORN", "false").lower() == "true":
        import gunicorn.app.base

        class StandaloneApplication(gunicorn.app.base.BaseApplication):
            def __init__(self, app, options=None):
                self.options = options or {}
                self.application = app
                super().__init__()

            def load_config(self):
                for key, value in self.options.items():
                    self.cfg.set(key.lower(), value)

            def load(self):
                return self.application

        options = {
            "bind": f"0.0.0.0:{port}",
            "workers": 1,
            "timeout": 300,
            "keepalive": 60,
        }
        StandaloneApplication(app, options).run()
    else:
        # 开发环境用 Flask 内置服务器
        app.run(host="0.0.0.0", port=port, threaded=True)
