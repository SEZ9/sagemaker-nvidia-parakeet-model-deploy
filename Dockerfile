# Parakeet ASR - SageMaker 自定义容器
# 基于 NVIDIA NeMo 官方镜像 (需要 CUDA 12.x, g6e 支持)
FROM nvcr.io/nvidia/nemo:24.07

# 环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/tmp/hf_cache
ENV TORCH_HOME=/tmp/torch_cache
ENV NEMO_CACHE_DIR=/tmp/nemo_cache

# 安装 Web 服务依赖
RUN pip install --no-cache-dir \
    flask>=3.0.0 \
    gunicorn>=21.0.0 \
    boto3>=1.35.0 \
    requests>=2.31.0 \
    soundfile>=0.12.1

# 创建目录
RUN mkdir -p /opt/ml/model /opt/ml/code /tmp/hf_cache /tmp/torch_cache /tmp/nemo_cache

# 注意: 模型会在首次启动时自动下载（构建时无 GPU 无法预下载）

# 复制推理代码
COPY serve.py /opt/ml/code/serve.py

# 工作目录
WORKDIR /opt/ml/code

# 暴露端口
EXPOSE 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8080/ping || exit 1

# 启动服务
ENTRYPOINT ["python", "/opt/ml/code/serve.py"]
