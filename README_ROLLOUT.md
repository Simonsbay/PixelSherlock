# Agent V3 Rollout Generator

This directory contains scripts for generating rollouts for RL (Reinforcement Learning) training using agent_v3's tools.

## Overview

The rollout generator combines agent_v3's specialized tools with a ReAct-style workflow to generate training data for RL:

- **image_multi_crop**: Crop important regions from images
- **image_recognize**: Search for similar images (visual search)
- **web_text_search**: Web search for text information

## Files

### 1. `generate_rollout.py` - Standalone Rollout Generator

A standalone script that generates rollouts without rLLM dependencies. Best for testing and generating individual rollouts.

**Usage:**

```bash
# Same input as main2.py
python generate_rollout.py --image_path test_data/location/test5.jpg --question "这是哪?"

# With ground truth answer
python generate_rollout.py --image_path test_data/location/test5.jpg --question "这是哪?" --answer "天安门"

# Custom output directory
python generate_rollout.py --image_path test.jpg --question "这是哪?" --output_dir ./my_rollouts

# Use different model
python generate_rollout.py --image_path test.jpg --question "这是哪?" --model qwen3-vl-72b-instruct

# Save full message history
python generate_rollout.py --image_path test.jpg --question "这是哪?" --format raw
```

**Output Format:**

By default, outputs in "rllm" format (optimized for training):
```json
{
  "task": {
    "question": "这是哪?",
    "answer": "天安门",
    "image_path": "/path/to/image.jpg"
  },
  "trajectory": {
    "steps": [...],
    "num_steps": 5
  },
  "prediction": "这是天安门",
  "termination": "answer",
  "is_correct": true,
  "metrics": {...}
}
```

### 2. `run_rollout.py` - Simple Executable Script

A simpler wrapper around the workflow. Same functionality as `generate_rollout.py` but with shorter argument names.

**Usage:**

```bash
# Basic usage
python run_rollout.py -i test_data/location/test5.jpg -q "这是哪?"

# With ground truth
python run_rollout.py -i test.jpg -q "这是哪?" -a "天安门"

# With custom output
python run_rollout.py -i test.jpg -q "这是哪?" -o ./rollouts

# Save full message history
python run_rollout.py -i test.jpg -q "这是哪?" --save_messages
```

### 3. `agent_v3_workflow.py` (in rllm/vision_deepresearch_async_workflow/)

rLLM-compatible workflow class that can be integrated into the training pipeline.

**Standalone Usage:**

```python
import asyncio
from agent_v3_workflow import AgentV3Workflow

async def main():
    workflow = AgentV3Workflow()

    task = {
        "question": "这是哪?",
        "images": ["/path/to/image.jpg"],
        "answer": "天安门",  # Optional
    }

    episode = await workflow.run(task, uid="task_001")

    print(f"Prediction: {episode.info['prediction']}")
    print(f"Answer: {episode.info['answer']}")
    print(f"Correct: {episode.is_correct}")

asyncio.run(main())
```

**rLLM Training Integration:**

```python
from rllm.trainer.agent_trainer import AgentTrainer
from vision_deepresearch_async_workflow.agent_v3_workflow import AgentV3Workflow

trainer = AgentTrainer(
    workflow_class=AgentV3Workflow,
    workflow_args={},
    config=config,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
)
trainer.train()
```

## Workflow

The rollout generation follows this workflow:

1. **Input**: Image + Question (e.g., "这是哪?")
2. **Phase 1 - Image Cropping**: LLM calls `image_multi_crop` to extract important regions
3. **Phase 2 - Visual Recognition**: LLM calls `image_recognize` on selected crops
4. **Phase 3 - Web Search**: LLM calls `web_text_search` to gather additional information
5. **Phase 4 - Final Answer**: LLM generates answer wrapped in `<answer>...</answer>` tags

## Rollout Structure

Each rollout contains:

```json
{
  "task": {
    "question": "用户问题",
    "answer": "标准答案（如果有）",
    "image_path": "图像路径"
  },
  "trajectory": {
    "steps": [
      {
        "model_response": "模型输出（包含tool_call或answer）",
        "observation": "工具返回结果",
        "has_error": false
      }
    ]
  },
  "prediction": "最终预测答案",
  "termination": "终止原因（answer/error/max_rounds_reached等）",
  "is_correct": true/false,
  "metrics": {
    "rounds": 5,
    "time_taken": 12.34,
    "prompt_tokens": 1234,
    "completion_tokens": 567
  }
}
```

## Environment Variables

- `DASHSCOPE_API_KEY`: DashScope API key for LLM calls

## Comparison with main2.py

| Aspect | main2.py | generate_rollout.py |
|--------|----------|---------------------|
| Purpose | Interactive inference | RL training data generation |
| Output | Console output | JSON rollout file |
| Format | Human-readable | Machine-readable (rLLM compatible) |
| Tools | Same (image_multi_crop, image_recognize, web_text_search) | Same |
| Workflow | ReAct loop | ReAct loop with step tracking |

## Testing

Test with the same images as main2.py:

```bash
cd /gemini-3/space/data/wangcy/Vision-DeepResearch-main/agent_v3

# Test with location images
python generate_rollout.py --image_path test_data/location/test5.jpg --question "这是哪?"

# Test with other images
python generate_rollout.py --image_path test_data/other/test1.jpg --question "这是什么?"
```

## Integration with rLLM Training

To use this workflow in rLLM training:

1. Ensure `agent_v3_workflow.py` is in your Python path
2. Use `AgentV3Workflow` as the workflow class in your trainer config
3. The workflow will automatically generate rollouts compatible with rLLM's Episode format

See `train_deepresearch_workflow_megatron.py` for a reference on how to set up the trainer.
