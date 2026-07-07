#!/usr/bin/env python3
"""
批量测试脚本 - 遍历文件夹中的所有图片并运行 generate_rollout

Usage:
    python batch_test.py --image_dir /path/to/images --question "这是哪?"
    python batch_test.py --image_dir /path/to/images --question "这是哪?" --answer_file answers.json
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_rollout import RolloutGenerator, format_rollout_for_rllm


async def process_single_image(
    generator: RolloutGenerator,
    image_path: str,
    question: str,
    answer: str = None,
    idx: int = 0,
    total: int = 0
) -> Dict[str, Any]:
    """处理单张图片。"""
    print(f"\n{'='*60}")
    print(f"[{idx+1}/{total}] Processing: {image_path}")
    print(f"{'='*60}")

    try:
        result = await generator.run(
            question=question,
            image_path=image_path,
            answer=answer,
        )
        return result
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return {
            "error": str(e),
            "question": question,
            "answer": answer,
            "image_path": image_path,
            "prediction": "",
            "termination": "error",
            "rounds": 0,
        }


def load_answer_file(answer_file: str) -> Dict[str, str]:
    """加载答案文件，支持 JSON 格式：{"image_name": "answer", ...}"""
    if not answer_file or not os.path.exists(answer_file):
        return {}

    with open(answer_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 支持两种格式：
    # 1. {"image_name.jpg": "answer", ...}
    # 2. [{"image": "name.jpg", "answer": "..."}, ...]
    if isinstance(data, dict):
        return data
    elif isinstance(data, list):
        return {item.get("image", item.get("image_path", "")): item.get("answer", "")
                for item in data}
    return {}


def get_image_files(image_dir: str, extensions: tuple = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')) -> List[str]:
    """获取文件夹中的所有图片文件。"""
    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"Error: Directory not found: {image_dir}")
        return []

    image_files = []
    for ext in extensions:
        image_files.extend(image_dir.glob(f"*{ext}"))
        image_files.extend(image_dir.glob(f"*{ext.upper()}"))

    # 去重并排序
    image_files = sorted(set([str(f) for f in image_files]))
    return image_files


async def main():
    parser = argparse.ArgumentParser(
        description="批量测试 - 遍历图片文件夹运行 generate_rollout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 基本用法
    python batch_test.py --image_dir /path/to/images --question "这是哪?"

    # 带答案文件（用于评估正确率）
    python batch_test.py --image_dir /path/to/images --question "这是哪?" --answer_file answers.json

    # 自定义输出文件
    python batch_test.py --image_dir /path/to/images --question "这是哪?" --output batch_results.json

    # 指定 vLLM 配置
    python batch_test.py --image_dir ./images --question "这是哪?" --base_url http://localhost:8000/v1
        """
    )

    parser.add_argument("--image_dir", type=str, required=True,
                        help="图片文件夹路径")
    parser.add_argument("--question", type=str, default="这是哪?",
                        help="问题文本 (default: '这是哪?')")
    parser.add_argument("--answer_file", type=str, default=None,
                        help="答案文件路径 (JSON格式，可选)")
    parser.add_argument("--output", type=str, default=None,
                        help="汇总结果输出文件路径 (default: batch_result_<timestamp>.json)")
    parser.add_argument("--format", type=str, choices=["raw", "rllm"], default="rllm",
                        help="输出格式: raw 或 rllm (default: rllm)")

    # vLLM 配置
    parser.add_argument("--model", type=str, default=None,
                        help="模型名称 (default: 使用 generate_rollout.py 中的 DEFAULT_MODEL)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key (default: EMPTY)")
    parser.add_argument("--base_url", type=str, default=None,
                        help="vLLM base URL (default: http://localhost:1999/v1)")

    args = parser.parse_args()

    # 获取所有图片
    image_files = get_image_files(args.image_dir)
    if not image_files:
        print(f"No image files found in: {args.image_dir}")
        return

    print(f"\nFound {len(image_files)} images in {args.image_dir}")
    for i, img in enumerate(image_files[:5]):
        print(f"  {i+1}. {os.path.basename(img)}")
    if len(image_files) > 5:
        print(f"  ... and {len(image_files) - 5} more")

    # 加载答案
    answers = load_answer_file(args.answer_file) if args.answer_file else {}
    if answers:
        print(f"\nLoaded {len(answers)} answers from {args.answer_file}")

    # 创建生成器
    generator = RolloutGenerator(
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url
    )

    # 批量处理
    start_time = time.time()
    results = []

    for i, image_path in enumerate(image_files):
        # 尝试从答案文件中获取答案
        image_name = os.path.basename(image_path)
        answer = answers.get(image_name) or answers.get(image_path) or answers.get(os.path.splitext(image_name)[0])

        result = await process_single_image(
            generator=generator,
            image_path=image_path,
            question=args.question,
            answer=answer,
            idx=i,
            total=len(image_files)
        )

        # 转换格式
        if args.format == "rllm":
            result = format_rollout_for_rllm(result)

        results.append(result)

    total_time = time.time() - start_time

    # 统计信息
    successful = sum(1 for r in results if r.get("termination") == "answer")
    errors = sum(1 for r in results if r.get("termination") == "error" or "error" in r)
    correct = sum(1 for r in results if r.get("is_correct") is True)
    total_with_answer = sum(1 for r in results if r.get("answer"))

    stats = {
        "total_images": len(image_files),
        "successful": successful,
        "errors": errors,
        "correct": correct,
        "total_with_gt": total_with_answer,
        "accuracy": correct / total_with_answer if total_with_answer > 0 else None,
        "total_time": total_time,
        "avg_time_per_image": total_time / len(image_files) if image_files else 0,
    }

    # 构建汇总结果
    summary = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "image_dir": args.image_dir,
            "question": args.question,
            "model": args.model or generator.model,
            "base_url": args.base_url or generator.base_url,
            "format": args.format,
        },
        "stats": stats,
        "results": results,
    }

    # 保存结果
    output_file = args.output
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"batch_result_{timestamp}.json"

    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 打印汇总
    print(f"\n{'='*60}")
    print("BATCH TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total images: {stats['total_images']}")
    print(f"Successful: {stats['successful']}")
    print(f"Errors: {stats['errors']}")
    if stats['total_with_gt'] > 0:
        print(f"Correct: {stats['correct']}/{stats['total_with_gt']} ({stats['accuracy']:.2%})")
    print(f"Total time: {stats['total_time']:.2f}s")
    print(f"Avg time per image: {stats['avg_time_per_image']:.2f}s")
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
