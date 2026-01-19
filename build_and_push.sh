#!/bin/bash
# Parakeet ASR - 构建并推送到 ECR
# =====================================

set -e

# 配置 - 根据需要修改
REGION="us-east-1"
ACCOUNT_ID=""
REPO_NAME="parakeet-asr"
IMAGE_TAG="latest"

# ECR 完整地址
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}"

echo "=========================================="
echo "Parakeet ASR - Docker 镜像构建"
echo "=========================================="
echo "Region: ${REGION}"
echo "Account: ${ACCOUNT_ID}"
echo "Repository: ${REPO_NAME}"
echo "ECR URI: ${ECR_URI}"
echo "=========================================="

# 1. 登录 ECR
echo ""
echo "Step 1: 登录 ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# 2. 创建 ECR 仓库（如果不存在）
echo ""
echo "Step 2: 创建 ECR 仓库..."
aws ecr describe-repositories --repository-names ${REPO_NAME} --region ${REGION} 2>/dev/null || \
    aws ecr create-repository --repository-name ${REPO_NAME} --region ${REGION}

# 3. 登录 NVIDIA NGC（拉取 NeMo 基础镜像）
echo ""
echo "Step 3: 登录 NVIDIA NGC (如果需要)..."
echo "如果没有 NGC 账号，请访问 https://ngc.nvidia.com 注册"
echo "然后运行: docker login nvcr.io"
echo ""

# 4. 构建镜像
echo ""
echo "Step 4: 构建 Docker 镜像..."
echo "这可能需要 10-20 分钟（首次下载 NeMo 基础镜像约 15GB）"
docker build --platform linux/amd64 -t ${REPO_NAME}:${IMAGE_TAG} .

# 5. 标记镜像
echo ""
echo "Step 5: 标记镜像..."
docker tag ${REPO_NAME}:${IMAGE_TAG} ${ECR_URI}:${IMAGE_TAG}

# 6. 推送到 ECR
echo ""
echo "Step 6: 推送到 ECR..."
echo "这可能需要几分钟..."
docker push ${ECR_URI}:${IMAGE_TAG}

echo ""
echo "=========================================="
echo "✅ 完成！"
echo "=========================================="
echo "镜像地址: ${ECR_URI}:${IMAGE_TAG}"
echo ""
echo "下一步: 运行部署脚本"
echo "  python deploy_parakeet.py"
echo "=========================================="
