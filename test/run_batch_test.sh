#!/bin/bash
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"
export no_proxy="localhost,127.0.0.1,0.0.0.0"
# 批量测试脚本

# 配置
IMAGE_DIR="${1:-/gemini-3/space/data/baiz/data/SFT_data_p1}"
QUESTION="${2:-这是哪?}"
OUTPUT_DIR="/gemini-3/space/data/baiz/agent_v3/batch_res"

# vLLM 配置
BASE_URL="${VLLM_BASE_URL:-http://localhost:1999/v1}"
API_KEY="${VLLM_API_KEY:-EMPTY}"
MODEL="${VLLM_MODEL:-/gemini-3/space/data/baiz/models/Vision-DeepResearch-8B}"

echo "========================================"
echo "Batch Test Configuration"
echo "========================================"
echo "Image directory: $IMAGE_DIR"
echo "Question: $QUESTION"
echo "Base URL: $BASE_URL"
echo "Model: $MODEL"
echo "Output directory: $OUTPUT_DIR"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行批量测试
python /gemini-3/space/data/baiz/agent_v3/batch_test.py \
    --image_dir "$IMAGE_DIR" \
    --question "$QUESTION" \
    --output "$OUTPUT_DIR/batch_result_$(date +%Y%m%d_%H%M%S).json" \
    --format rllm \
    --base_url "$BASE_URL" \
    --api_key "$API_KEY" \
    --model "$MODEL"

echo ""
echo "Batch test completed!"
