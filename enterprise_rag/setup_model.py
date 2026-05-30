"""首次启动前运行此脚本，下载 Embedding 模型。

使用方法:
    python setup_model.py

模型信息:
    BAAI/bge-small-zh-v1.5  (~95 MB)
    用途: 中文文本语义向量化
    下载源: HuggingFace 镜像 (hf-mirror.com)
"""

import sys
from pathlib import Path

import requests

MODEL = "BAAI/bge-small-zh-v1.5"
MIRROR = "https://hf-mirror.com"
MODEL_DIR = Path(__file__).parent / "models" / "bge-small-zh-v1.5"


def main() -> None:
    if MODEL_DIR.exists() and (MODEL_DIR / "pytorch_model.bin").exists():
        print(f"模型已存在于: {MODEL_DIR}")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"正在从 {MIRROR} 下载模型 {MODEL} ...")
    print(f"目标目录: {MODEL_DIR}")

    try:
        # 获取文件列表
        api_url = f"{MIRROR}/api/models/{MODEL}"
        r = requests.get(api_url, timeout=30)
        r.raise_for_status()

        r = requests.get(
            f"{MIRROR}/api/models/{MODEL}/tree/main?recursive=True",
            timeout=30,
        )
        r.raise_for_status()
        all_files = [
            f["path"]
            for f in r.json()
            if ".git" not in f["path"] and "onnx/" not in f["path"] and "openvino/" not in f["path"]
        ]

        print(f"共 {len(all_files)} 个文件待下载")

        for fname in all_files:
            url = f"{MIRROR}/{MODEL}/resolve/main/{fname}"
            dest = MODEL_DIR / fname
            dest.parent.mkdir(parents=True, exist_ok=True)

            print(f"  下载: {fname} ...", end=" ", flush=True)
            r = requests.get(url, allow_redirects=True, timeout=120)
            if r.status_code == 200:
                dest.write_bytes(r.content)
                size_kb = len(r.content) // 1024
                print(f"OK ({size_kb} KB)")
            else:
                print(f"失败 HTTP {r.status_code}")

        print(f"\n模型下载完成: {MODEL_DIR}")

    except requests.exceptions.RequestException as exc:
        print(f"\n下载失败: {exc}", file=sys.stderr)
        print("请检查网络连接，或手动从 HuggingFace 下载模型放置到:", MODEL_DIR, file=sys.stderr)
        sys.exit(1)

    # 验证
    required = ["config.json", "pytorch_model.bin", "tokenizer.json", "vocab.txt"]
    missing = [f for f in required if not (MODEL_DIR / f).exists()]
    if missing:
        print(f"警告: 缺少文件 {missing}，模型可能不完整", file=sys.stderr)
    else:
        print("模型验证通过，可以启动应用。")


if __name__ == "__main__":
    main()
