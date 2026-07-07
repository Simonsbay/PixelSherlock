#!/usr/bin/env python3
"""
Parquet 预览转换工具
将 Parquet 文件转换为便于 VS Code 预览的格式（JSON/CSV/Markdown）

Usage:
    python parquet_preview.py /path/to/data.parquet
    python parquet_preview.py /path/to/data.parquet --format json --output preview.json
    python parquet_preview.py /path/to/data.parquet --format csv --head 100
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Install with: pip install pandas pyarrow")
    sys.exit(1)


def convert_to_json(df: pd.DataFrame, output_path: str, indent: int = 4, media_dir: str = None):
    """转换为 JSON 格式，二进制数据保存到单独文件"""
    import base64
    import hashlib

    output_path = Path(output_path)

    # 确定媒体文件保存目录
    if media_dir is None:
        media_dir = output_path.parent / (output_path.stem + "_media")
    else:
        media_dir = Path(media_dir)

    # 创建媒体目录
    media_dir.mkdir(parents=True, exist_ok=True)

    # 用于跟踪已保存的文件（避免重复保存相同内容）
    saved_hashes = {}

    def save_binary_data(data: bytes, row_idx: int, col_name: str, item_idx: int = None) -> str:
        """保存二进制数据到文件，返回相对路径"""
        # 计算数据哈希用于去重
        data_hash = hashlib.md5(data).hexdigest()

        if data_hash in saved_hashes:
            return saved_hashes[data_hash]

        # 尝试检测文件类型
        ext = ".bin"
        if data.startswith(b'\xff\xd8'):
            ext = ".jpg"
        elif data.startswith(b'\x89PNG'):
            ext = ".png"
        elif data.startswith(b'GIF89a') or data.startswith(b'GIF87a'):
            ext = ".gif"
        elif data.startswith(b'RIFF') and data[8:12] == b'WEBP':
            ext = ".webp"
        elif data.startswith(b'\x00\x00\x00 ftyp'):
            ext = ".mp4"

        # 生成文件名
        if item_idx is not None:
            filename = f"row{row_idx}_{col_name}_{item_idx}{ext}"
        else:
            filename = f"row{row_idx}_{col_name}{ext}"

        file_path = media_dir / filename

        # 如果文件已存在，添加哈希前缀避免冲突
        if file_path.exists():
            filename = f"{data_hash[:8]}_{filename}"
            file_path = media_dir / filename

        # 保存文件
        with open(file_path, 'wb') as f:
            f.write(data)

        # 返回相对路径（相对于 JSON 文件）
        rel_path = f"./{media_dir.name}/{filename}"
        saved_hashes[data_hash] = rel_path
        return rel_path

    def convert_to_serializable(obj, depth=0, max_depth=100, row_idx=None, col_name=None):
        """深度转换所有不可序列化类型"""
        if depth > max_depth:
            return str(obj)

        if isinstance(obj, bytes):
            # 二进制数据保存到文件
            return save_binary_data(obj, row_idx, col_name)

        if hasattr(obj, 'tolist') and callable(getattr(obj, 'tolist')):
            return convert_to_serializable(obj.tolist(), depth + 1, max_depth, row_idx, col_name)

        if hasattr(obj, 'item') and callable(getattr(obj, 'item')):
            return convert_to_serializable(obj.item(), depth + 1, max_depth, row_idx, col_name)

        if isinstance(obj, dict):
            return {str(k): convert_to_serializable(v, depth + 1, max_depth, row_idx, str(k)) for k, v in obj.items()}

        if isinstance(obj, (list, tuple)):
            result = []
            for i, v in enumerate(obj):
                # 对于列表中的元素，尝试传递列名信息（如果是图片数组）
                item_col_name = f"{col_name}_{i}" if col_name else str(i)
                result.append(convert_to_serializable(v, depth + 1, max_depth, row_idx, item_col_name))
            return result

        # 处理其他常见不可序列化类型
        if isinstance(obj, (set, frozenset)):
            return [convert_to_serializable(v, depth + 1, max_depth, row_idx, col_name) for v in obj]

        return obj

    # 逐行转换，确保完全序列化
    records = []
    for row_idx, (_, row) in enumerate(df.iterrows()):
        record = convert_to_serializable(row.to_dict(), row_idx=row_idx)
        records.append(record)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=indent)

    print(f"✅ JSON  saved: {output_path} ({len(records)} records)")
    print(f"📁 Media saved: {media_dir} ({len(saved_hashes)} unique files)")


def convert_to_csv(df: pd.DataFrame, output_path: str):
    """转换为 CSV 格式"""
    df.to_csv(output_path, index=False, encoding='utf-8-sig')  # BOM for Excel compatibility
    print(f"✅ CSV   saved: {output_path} ({len(df)} records, {len(df.columns)} columns)")


def convert_to_markdown(df: pd.DataFrame, output_path: str, max_rows: int = 50):
    """转换为 Markdown 表格格式（适合小数据量预览）"""
    preview_df = df.head(max_rows)

    lines = []
    lines.append(f"# Parquet Preview ({len(preview_df)} of {len(df)} rows)\n")

    # 列信息
    lines.append("## Schema\n")
    lines.append("| Column | Type | Non-Null Count |")
    lines.append("|--------|------|----------------|")
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = df[col].notna().sum()
        lines.append(f"| {col} | {dtype} | {non_null}/{len(df)} |")
    lines.append("")

    # 数据预览
    lines.append(f"## Data Preview (first {len(preview_df)} rows)\n")

    # 构建 markdown 表格
    headers = "| " + " | ".join(str(c) for c in preview_df.columns) + " |"
    separator = "|" + "|".join([" --- " for _ in preview_df.columns]) + "|"

    lines.append(headers)
    lines.append(separator)

    for _, row in preview_df.iterrows():
        values = []
        for val in row:
            # 截断长文本
            text = str(val) if val is not None else "NULL"
            text = text.replace("|", "\\|").replace("\n", " ")
            if len(text) > 100:
                text = text[:97] + "..."
            values.append(text)
        lines.append("| " + " | ".join(values) + " |")

    lines.append("")

    # 统计信息
    lines.append("## Statistics\n")
    lines.append(f"- **Total Rows**: {len(df)}")
    lines.append(f"- **Total Columns**: {len(df.columns)}")
    lines.append(f"- **Memory Usage**: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    lines.append("")

    # 列名列表
    lines.append("## Columns\n")
    for i, col in enumerate(df.columns, 1):
        lines.append(f"{i}. `{col}` ({df[col].dtype})")
    lines.append("")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    print(f"✅ Markdown saved: {output_path} (preview of {len(preview_df)} rows)")


def print_summary(df: pd.DataFrame, filepath: str):
    """打印数据摘要"""
    print("\n" + "=" * 60)
    print(f"📊 Parquet File: {filepath}")
    print("=" * 60)
    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Memory:  {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    print("\n📋 Column Schema:")
    print("-" * 60)

    for i, col in enumerate(df.columns, 1):
        dtype = str(df[col].dtype)
        non_null = df[col].notna().sum()
        null_pct = (len(df) - non_null) / len(df) * 100
        null_info = f"{null_pct:.1f}% null" if null_pct > 0 else "no null"
        print(f"  {i:2d}. {col:30s} | {dtype:15s} | {null_info}")

    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Parquet to preview-friendly formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 默认输出：打印摘要 + 生成同名 JSON（二进制数据保存到 data_media 目录）
    python parquet_preview.py data.parquet

    # 指定格式
    python parquet_preview.py data.parquet --format csv

    # 只预览前 N 行
    python parquet_preview.py data.parquet --head 100 --format json

    # 自定义输出路径
    python parquet_preview.py data.parquet --output ./preview/my_data.json

    # 查看列统计
    python parquet_preview.py data.parquet --summary-only

    # 自定义媒体文件保存目录
    python parquet_preview.py data.parquet --media-dir ./images
        """
    )

    parser.add_argument("parquet_file", type=str,default="/gemini-3/space/data/baiz/data/simpleVQA/data/test-00000-of-00001.parquet",
                        help="输入的 Parquet 文件路径")
    parser.add_argument("--format", type=str, choices=["json", "csv", "markdown", "md"],
                        default="json",
                        help="输出格式 (default: json)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径 (默认: 与输入文件同名，仅扩展名不同)")
    parser.add_argument("--head", type=int, default=None,
                        help="只转换前 N 行 (default: 全部)")
    parser.add_argument("--summary-only", action="store_true",
                        help="只打印摘要，不生成文件")
    parser.add_argument("--indent", type=int, default=4,
                        help="JSON 缩进空格数 (default: 4)")
    parser.add_argument("--media-dir", type=str, default=None,
                        help="媒体文件保存目录 (default: 与输出文件同名目录 + \"_media\")")

    args = parser.parse_args()

    # 检查文件
    if not os.path.exists(args.parquet_file):
        print(f"❌ Error: File not found: {args.parquet_file}")
        sys.exit(1)

    # 读取 parquet
    try:
        print(f"📖 Reading {args.parquet_file}...")
        df = pd.read_parquet(args.parquet_file)
    except Exception as e:
        print(f"❌ Error reading parquet: {e}")
        sys.exit(1)

    # 截取前 N 行
    if args.head:
        df = df.head(args.head)
        print(f"📌 Using first {args.head} rows only\n")

    # 打印摘要
    print_summary(df, args.parquet_file)

    if args.summary_only:
        return

    # 确定输出路径
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.parquet_file)
        ext = "md" if args.format in ["markdown", "md"] else args.format
        output_path = input_path.with_suffix(f".{ext}")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # 转换
    print(f"\n📝 Converting to {args.format.upper()}...")

    if args.format == "json":
        convert_to_json(df, str(output_path), args.indent, args.media_dir)
    elif args.format == "csv":
        convert_to_csv(df, str(output_path))
    elif args.format in ["markdown", "md"]:
        convert_to_markdown(df, str(output_path), max_rows=args.head or 50)

    print(f"\n💡 Tip: 在 VS Code 中打开查看: code '{output_path}'")


if __name__ == "__main__":
    main()
