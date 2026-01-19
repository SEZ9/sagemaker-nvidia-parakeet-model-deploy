#!/usr/bin/env python3
"""
Parakeet ASR - SageMaker 部署脚本
"""

import boto3
from datetime import datetime

# ====== 配置 ======
REGION = "us-east-1"
ACCOUNT_ID = ""
ROLE = "arn:aws:iam::XXXXXX:role/service-role/AmazonSageMaker-ExecutionRole-20221214T123867"
INSTANCE_TYPE = "ml.g6e.12xlarge"  # 单卡 A10G，足够跑 0.6B 模型

# ECR 镜像地址
REPO_NAME = "parakeet-asr"
IMAGE_URI = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{REPO_NAME}:latest"


def deploy():
    """部署 Parakeet 到 SageMaker"""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_name = f"parakeet-tdt-{timestamp}"
    endpoint_config_name = f"parakeet-config-{timestamp}"
    endpoint_name = f"parakeet-asr-{timestamp}"

    sm = boto3.client("sagemaker", region_name=REGION)

    print("=" * 50)
    print("Parakeet ASR - SageMaker 部署")
    print("=" * 50)
    print(f"Region: {REGION}")
    print(f"Instance: {INSTANCE_TYPE}")
    print(f"Image: {IMAGE_URI}")
    print(f"Model name: {model_name}")
    print(f"Endpoint: {endpoint_name}")
    print("=" * 50)

    # 1. 创建模型
    print("\nStep 1: 创建 SageMaker Model...")
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": IMAGE_URI,
            "Mode": "SingleModel",
            "Environment": {
                "SAGEMAKER_BIND_TO_PORT": "8080",
            }
        },
        ExecutionRoleArn=ROLE,
    )
    print(f"✓ Model created: {model_name}")

    # 2. 创建端点配置
    print("\nStep 2: 创建 Endpoint Config...")
    sm.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[{
            "VariantName": "primary",
            "ModelName": model_name,
            "InstanceType": INSTANCE_TYPE,
            "InitialInstanceCount": 1,
            "ContainerStartupHealthCheckTimeoutInSeconds": 600,  # 10分钟启动超时
        }]
    )
    print(f"✓ Endpoint config created: {endpoint_config_name}")

    # 3. 创建端点
    print("\nStep 3: 创建 Endpoint...")
    print("这需要 5-10 分钟，请耐心等待...")
    sm.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name,
    )

    # 等待端点就绪
    print("等待端点状态变为 InService...")
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(
        EndpointName=endpoint_name,
        WaiterConfig={"Delay": 30, "MaxAttempts": 60}
    )

    print("\n" + "=" * 50)
    print("✅ 部署完成！")
    print("=" * 50)
    print(f"Endpoint name: {endpoint_name}")
    print(f"\n测试命令:")
    print(f"  修改 test_parakeet.py 中的 ENDPOINT_NAME = \"{endpoint_name}\"")
    print(f"  然后运行: python test_parakeet.py")
    print("=" * 50)

    return endpoint_name


def cleanup(endpoint_name: str):
    """清理资源"""
    sm = boto3.client("sagemaker", region_name=REGION)

    print(f"清理端点: {endpoint_name}")

    # 获取关联资源
    try:
        endpoint_info = sm.describe_endpoint(EndpointName=endpoint_name)
        config_name = endpoint_info["EndpointConfigName"]

        config_info = sm.describe_endpoint_config(EndpointConfigName=config_name)
        model_name = config_info["ProductionVariants"][0]["ModelName"]

        # 删除端点
        print(f"删除 Endpoint: {endpoint_name}")
        sm.delete_endpoint(EndpointName=endpoint_name)

        # 等待端点删除
        print("等待端点删除...")
        waiter = sm.get_waiter("endpoint_deleted")
        waiter.wait(EndpointName=endpoint_name)

        # 删除端点配置
        print(f"删除 Endpoint Config: {config_name}")
        sm.delete_endpoint_config(EndpointConfigName=config_name)

        # 删除模型
        print(f"删除 Model: {model_name}")
        sm.delete_model(ModelName=model_name)

        print("✅ 清理完成")

    except Exception as e:
        print(f"清理出错: {e}")


def list_endpoints():
    """列出所有 Parakeet 端点"""
    sm = boto3.client("sagemaker", region_name=REGION)

    endpoints = sm.list_endpoints(
        NameContains="parakeet",
        StatusEquals="InService"
    )

    print("当前运行的 Parakeet 端点:")
    for ep in endpoints["Endpoints"]:
        print(f"  - {ep['EndpointName']} ({ep['EndpointStatus']})")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "cleanup" and len(sys.argv) > 2:
            cleanup(sys.argv[2])
        elif sys.argv[1] == "list":
            list_endpoints()
        else:
            print("用法:")
            print("  python deploy_parakeet.py          # 部署")
            print("  python deploy_parakeet.py list     # 列出端点")
            print("  python deploy_parakeet.py cleanup <endpoint_name>  # 清理")
    else:
        deploy()
