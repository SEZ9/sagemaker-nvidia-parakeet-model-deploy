#!/usr/bin/env python3
"""
Parakeet ASR - 测试脚本
"""

import boto3
import json
import time
import base64
import statistics
import os

# ====== 配置 ======
REGION = "us-east-1"
ENDPOINT_NAME = ""  # TODO: 修改为实际的 endpoint 名称

# 音频文件
AUDIO_FILE = "test.wav"

sm_runtime = boto3.client("sagemaker-runtime", region_name=REGION)


def test_transcription(num_tests=20):
    """测试转录性能"""

    # 读取音频
    if not os.path.exists(AUDIO_FILE):
        print(f"错误: 找不到 {AUDIO_FILE}")
        return

    with open(AUDIO_FILE, "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    file_size = len(audio_bytes)
    print(f"音频文件: {AUDIO_FILE}")
    print(f"文件大小: {file_size} bytes ({file_size/1024/1024:.2f} MB)")

    # 构造请求
    payload = {
        "audio": {"data": audio_base64},
        "timestamps": False
    }

    print(f"\n端点: {ENDPOINT_NAME}")
    print(f"开始性能测试 ({num_tests} 次)...\n")

    # 性能测试
    test_times = []
    server_times = []

    for i in range(num_tests):
        start_time = time.time()

        try:
            resp = sm_runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType="application/json",
                Body=json.dumps(payload)
            )

            result = json.loads(resp["Body"].read().decode("utf-8"))
            duration = time.time() - start_time
            test_times.append(duration)
            print(result)
            server_time = result.get("processing_time", 0)
            server_times.append(server_time)

            print(f"测试 {i+1}/{num_tests}: 总耗时 {duration:.2f}s, 服务端 {server_time}s")

            if i == 0:
                text = result.get("text", "")
                print(f"转录结果: {text[:200]}...")
                if "error" in result:
                    print(f"错误: {result['error']}")
                    return

        except Exception as e:
            print(f"测试 {i+1} 失败: {e}")
            continue

    # 统计
    if test_times:
        print("\n" + "=" * 50)
        print("性能统计 (Parakeet TDT 0.6B)")
        print("=" * 50)
        print(f"测试次数: {len(test_times)}")
        print(f"\n端到端延迟:")
        print(f"  平均: {statistics.mean(test_times):.2f}s")
        print(f"  最短: {min(test_times):.2f}s")
        print(f"  最长: {max(test_times):.2f}s")
        print(f"  中位数: {statistics.median(test_times):.2f}s")

        if server_times:
            print(f"\n服务端处理时间:")
            print(f"  平均: {statistics.mean(server_times):.3f}s")
            print(f"  最短: {min(server_times):.3f}s")
            print(f"  最长: {max(server_times):.3f}s")

        if len(test_times) > 1:
            print(f"\n标准差: {statistics.stdev(test_times):.2f}s")


def test_with_timestamps():
    """测试词级时间戳"""

    if not os.path.exists(AUDIO_FILE):
        print(f"错误: 找不到 {AUDIO_FILE}")
        return

    with open(AUDIO_FILE, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "audio": {"data": audio_base64},
        "timestamps": True
    }

    print("测试词级时间戳...\n")

    resp = sm_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload)
    )

    result = json.loads(resp["Body"].read().decode("utf-8"))

    print(f"转录文本: {result.get('text', '')}\n")

    timestamps = result.get("timestamps", {})
    if "word" in timestamps and timestamps["word"]:
        print("词级时间戳 (前10个):")
        for w in timestamps["word"][:10]:
            print(f"  [{w['start']:.2f}s - {w['end']:.2f}s] {w['word']}")
    else:
        print("未返回时间戳数据")


def test_with_url(audio_url: str):
    """测试 URL 输入"""

    payload = {
        "audio": {"path": audio_url},
        "timestamps": False
    }

    print(f"测试 URL 输入: {audio_url}\n")

    start_time = time.time()
    resp = sm_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload)
    )

    result = json.loads(resp["Body"].read().decode("utf-8"))
    duration = time.time() - start_time

    print(f"转录结果: {result.get('text', '')}")
    print(f"总耗时: {duration:.2f}s")
    print(f"服务端处理: {result.get('processing_time', 'N/A')}s")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "timestamps":
            test_with_timestamps()
        elif sys.argv[1] == "url" and len(sys.argv) > 2:
            test_with_url(sys.argv[2])
        else:
            print("用法:")
            print("  python test_parakeet.py              # 性能测试")
            print("  python test_parakeet.py timestamps   # 测试时间戳")
            print("  python test_parakeet.py url <url>    # 测试URL输入")
    else:
        test_transcription()
