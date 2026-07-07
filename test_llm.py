from openai import OpenAI
# 设置 OpenAI 的 API 密钥和 API 基础 URL 以使用 vLLM 的 API 服务器。

import pandas as pd
df = pd.read_parquet('/gemini-3/space/data/baiz/data/vision-deepresearch_SFT_Demo_1k.parquet')  # 仅读取 0 行
print("Columns:", df.columns.tolist())

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:1999/v1"
print("api地址",openai_api_base)

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


chat_response = client.chat.completions.create(
    model="/gemini-3/space/data/baiz/checkpoints/Qwen3-VL-30B-SFT-364-lr2e5-2ep/v19-20260309-175802/checkpoint-91",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ]
)
print("Chat response:", chat_response)