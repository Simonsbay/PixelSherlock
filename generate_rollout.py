"""
Agent V3 Rollout Generator - Standalone Executable Script

This script generates rollouts for RL training using agent_v3 tools.
It accepts the same input format as main2.py.

Usage:
    python generate_rollout.py --image_path test_data/location/test5.jpg --question "这是哪?"
    python generate_rollout.py --image_path test_data/location/test5.jpg --question "这是哪?" --answer "天安门"

Output:
    - Saves rollout JSON to ./rollouts/ directory
    - Prints rollout summary to console
"""

import asyncio
import json
import os
import sys
import time
import base64
import re
from datetime import datetime
from typing import Any, List, Optional, Dict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from openai import AsyncOpenAI

# Import agent_v3 modules
from prompts import LOCATION_PROMPT

# Delayed imports to avoid circular import:
# image_multi_crop -> qwen_agent -> qwen_agent.tools -> image_multi_crop
# These will be imported in __init__ method

# vLLM API Configuration
VLLM_API_KEY = 'EMPTY'  # vLLM serve 默认不需要 API key，或设置为自定义 token
VLLM_BASE_URL = 'http://localhost:1999/v1'  # vLLM serve 默认地址
DEFAULT_MODEL = "/gemini-3/space/data/baiz/checkpoints/Qwen3-VL-30B-SFT-364-lr2e5-2ep/v19-20260309-175802/checkpoint-91"  # 修改为 vLLM 加载的模型名称

# Workflow Configuration
MAX_LLM_CALL_PER_RUN = 50
MAX_PROMPT_LENGTH_PER_RUN = 64000
MAX_RESPONSE_LENGTH_PER_RUN = 4096


class RolloutGenerator:
    """
    Generates rollouts for RL training using agent_v3 tools.

    This class implements a ReAct-style workflow similar to rLLM's DeepResearchAgent
    but uses agent_v3's specific tools (image_multi_crop, image_recognize, web_search).
    """

    def __init__(self, api_key: str = None, model: str = None, base_url: str = None):
        self.api_key = api_key or VLLM_API_KEY
        self.model = model or DEFAULT_MODEL
        self.base_url = base_url or VLLM_BASE_URL

        # Delayed import to avoid circular import
        from image_multi_crop import ImageMultiCrop
        from image_recognize2 import ImageRecognize
        from text_search import WebSearch

        # Initialize tools
        self.tools = {
            "image_multi_crop": ImageMultiCrop(),
            "image_recognize": ImageRecognize(),
            "web_search": WebSearch(),
        }

        # Token tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def reset(self):
        """Reset token counters."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    async def call_llm(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """
        Call vLLM serve API (OpenAI-compatible).

        Args:
            messages: List of message dictionaries
            **kwargs: Additional arguments for the API call

        Returns:
            Dictionary with content and token usage
        """
        async with AsyncOpenAI(api_key=self.api_key, base_url=self.base_url) as client:
            # Convert messages to OpenAI format with image support
            openai_messages = []
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")

                if role == "system":
                    openai_messages.append({"role": "system", "content": content})

                elif role == "user":
                    # Check if content contains image reference
                    img_match = re.search(r'<image>(.*?)</image>\n?', content)
                    if img_match:
                        img_path = img_match.group(1)
                        text = re.sub(r'<image>.*?</image>\n?', "", content).strip()

                        # Read and encode image
                        with open(img_path, "rb") as f:
                            img_base64 = base64.b64encode(f.read()).decode("utf-8")

                        openai_messages.append({
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                                {"type": "text", "text": text},
                            ]
                        })
                    else:
                        openai_messages.append({"role": "user", "content": content})

                elif role == "assistant":
                    openai_messages.append({"role": "assistant", "content": content})

            completion = await client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                stream=False,
                max_tokens=kwargs.get("max_tokens", MAX_RESPONSE_LENGTH_PER_RUN),
                #temperature=kwargs.get("temperature", 0.01),
                temperature=0.01,
            )

            response_content = completion.choices[0].message.content

            # Track tokens
            if completion.usage:
                self.total_prompt_tokens += completion.usage.prompt_tokens
                self.total_completion_tokens += completion.usage.completion_tokens

            return {
                "content": response_content,
                "prompt_tokens": completion.usage.prompt_tokens if completion.usage else 0,
                "completion_tokens": completion.usage.completion_tokens if completion.usage else 0,
            }

    async def execute_tool(self, tool_name: str, tool_args: Dict, messages: List[Dict] = None) -> str:
        """
        Execute a tool with the given arguments.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            messages: Context messages (for some tools)

        Returns:
            Tool execution result as string
        """
        if tool_name not in self.tools:
            return f"Tool {tool_name} not found. Available: {list(self.tools.keys())}"

        tool = self.tools[tool_name]

        try:
            if tool_name == "image_multi_crop":
                # Convert messages to qwen_agent Message format
                from qwen_agent.llm.schema import Message, ContentItem
                qwen_messages = []
                for msg in messages:
                    role = msg.get("role")
                    content = msg.get("content", "")
                    if role == "user" and "<image>" in content:
                        # Extract image path and text
                        import re
                        img_match = re.search(r'<image>(.*?)</image>', content)
                        if img_match:
                            img_path = img_match.group(1)
                            text = re.sub(r'<image>.*?</image>\n?', "", content).strip()
                            qwen_content = [ContentItem(image=img_path)]
                            if text:
                                qwen_content.append(ContentItem(text=text))
                            qwen_messages.append(Message(role="user", content=qwen_content))
                        else:
                            qwen_messages.append(Message(role="user", content=content))
                    elif role == "assistant":
                        qwen_messages.append(Message(role="assistant", content=content))
                    elif role == "system":
                        qwen_messages.append(Message(role="system", content=content))
                    else:
                        qwen_messages.append(Message(role=role, content=content))

                result = tool.call(tool_args, messages=qwen_messages)
                # Format result - handle ContentItem with text or image
                output_lines = []
                for item in result:
                    text = getattr(item, 'text', None)
                    if text is not None:
                        output_lines.append(str(text))
                    else:
                        image = getattr(item, 'image', None)
                        if image is not None:
                            output_lines.append(f"[Image saved: {image}]")
                return "\n".join(output_lines) if output_lines else "No crops generated"

            elif tool_name == "image_recognize":
                # Convert messages to qwen_agent Message format
                from qwen_agent.llm.schema import Message, ContentItem
                qwen_messages = []
                for msg in messages:
                    role = msg.get("role")
                    content = msg.get("content", "")
                    if role == "user" and "<image>" in content:
                        import re
                        img_match = re.search(r'<image>(.*?)</image>', content)
                        if img_match:
                            img_path = img_match.group(1)
                            text = re.sub(r'<image>.*?</image>\n?', "", content).strip()
                            qwen_content = [ContentItem(image=img_path)]
                            if text:
                                qwen_content.append(ContentItem(text=text))
                            qwen_messages.append(Message(role="user", content=qwen_content))
                        else:
                            qwen_messages.append(Message(role="user", content=content))
                    elif role == "assistant":
                        qwen_messages.append(Message(role="assistant", content=content))
                    elif role == "system":
                        qwen_messages.append(Message(role="system", content=content))
                    else:
                        qwen_messages.append(Message(role=role, content=content))

                # Extract question from messages for search
                user_question = ""
                for msg in qwen_messages:
                    if msg.role == "user":
                        if isinstance(msg.content, list):
                            for item in msg.content:
                                if hasattr(item, 'text') and item.text:
                                    user_question = item.text
                                    break
                        elif isinstance(msg.content, str):
                            user_question = msg.content
                        if user_question:
                            break

                # Call async search directly instead of call() to avoid asyncio.run() conflict
                sub_image_info_list = tool_args.get('sub_image_info_list', [])
                result = await tool.search(sub_image_info_list, user_question)
                if not result:
                    return 'The ImageRecognize tool could not find relevant reference information'
                if isinstance(result, dict):
                    return json.dumps(result, ensure_ascii=False, indent=2)
                return str(result)

            elif tool_name == "web_search":
                # Call async search directly instead of call() to avoid asyncio.run() conflict
                query = tool_args.get('query', '')
                result = await tool.search(query)
                if not result:
                    return 'The WebSearch tool could not find relevant reference information'
                if isinstance(result, list):
                    output_lines = []
                    for item in result:
                        text = getattr(item, 'text', None)
                        if text is not None:
                            output_lines.append(str(text))
                        else:
                            output_lines.append(str(item))
                    return "\n".join(output_lines) if output_lines else "No search results"
                return str(result)

            else:
                return f"Unknown tool: {tool_name}"

        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    async def run(self, question: str, image_path: str, answer: str = None) -> Dict[str, Any]:
        """
        Run the full rollout generation workflow.

        This implements a ReAct-style loop:
        1. User provides image + question
        2. LLM thinks and decides to call tools (image_multi_crop -> image_recognize -> web_search)
        3. Tools execute and return observations
        4. LLM generates final answer in <answer> tags

        Args:
            question: User's question about the image
            image_path: Path to the image file
            answer: Ground truth answer (optional, for evaluation)

        Returns:
            Dictionary containing the full rollout data
        """
        self.reset()
        start_time = time.time()

        # Verify image exists
        if not os.path.exists(image_path):
            return {
                "error": f"Image not found: {image_path}",
                "question": question,
                "answer": answer,
                "messages": [],
                "prediction": "",
                "termination": "error",
                "rounds": 0,
            }

        # Build initial conversation
        # system_content = LOCATION_PROMPT + "\n\nRemember:\n"
        # system_content += "1. Think step by step and show your reasoning in <think>...</think> tags\n"
        # system_content += '2. Call tools using <tool_call>{"name": "...", "arguments": {...}}</tool_call>\n'
        # system_content += "3. When ready to answer, wrap your final response in <answer>...</answer> tags\n"
        system_content = LOCATION_PROMPT
        system_content +="\n\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"image_multi_crop\", \"description\": \"图像裁剪工具，根据用户的要求一次性对图片进行裁剪获得多张子图。\", \"parameters\": {\"type\": \"object\", \"properties\": {\"img_idx\": {\"type\": \"number\", \"description\": \"输入图像的索引\"}, \"crops\": {\"type\": \"array\", \"items\": {\"type\": \"object\", \"properties\": {\"bbox_2d\": {\"type\": \"array\", \"items\": {\"type\": \"number\"}, \"minItems\": 4, \"maxItems\": 4, \"description\": \"Bounding box as [x1, y1, x2, y2] in pixels or normalized\"}, \"label\": {\"type\": \"string\", \"description\": \"Label for this cropped region\"}, \"sub_idx\": {\"type\": \"number\", \"description\": \"Optional index to distinguish multiple crops (starting from 0)\"}}, \"required\": [\"bbox_2d\", \"label\", \"sub_idx\"]}, \"description\": \"List of regions to crop from the image\"}}, \"required\": [\"img_idx\", \"crops\"]}}}\n{\"type\": \"function\", \"function\": {\"name\": \"image_recognize\", \"description\": \"Image search engine, input the image and search for similar images with image information.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"sub_image_info_list\": {\"type\": \"array\", \"description\": \"筛选出来的子图列表，示例：[{\"selected_subject\": \"匹配主体的label\", \"selected_sub_idx\": \"子图编号\", \"reason\": \"说明选择原因\", \"sub_image_path\": \"子图路径\"}]\", \"items\": {\"type\": \"object\", \"properties\": {\"selected_subject\": {\"type\": \"string\", \"description\": \"匹配主体的label\"}, \"selected_sub_idx\": {\"type\": \"integer\", \"description\": \"子图编号（从0开始）\"}, \"reason\": {\"type\": \"string\", \"description\": \"说明选择原因\"}, \"sub_image_path\": {\"type\": \"string\", \"description\": \"子图路径\"}}, \"required\": [\"selected_subject\", \"selected_sub_idx\", \"reason\", \"sub_image_path\"]}}}, \"required\": [\"sub_image_info_list\"]}}}\n{\"type\": \"function\", \"function\": {\"name\": \"web_search\", \"description\": \"网络搜索工具。当通过图像工具识别出视觉主体以及相关信息后，需要使用此工具联网搜索补充信息\", \"parameters\": {\"type\": \"object\", \"properties\": {\"query\": {\"type\": \"string\", \"description\": \"基于对话上下文生成的具体查询问题，需满足以下要求：\n1. 完整覆盖上下文待解答的核心需求，不遗漏关键信息；\n2. 表述清晰、简洁，符合搜索引擎的查询习惯；\n3. 避免模糊、宽泛的表述，尽可能包含关键限定词（如时间、地点、主体、具体维度等）；\n4. 以自然语言问题形式呈现（而非关键词堆砌），例如：\"2025年人工智能在医疗行业的应用前景\",\"多智能体协作在自动驾驶中的优势\"。\"}}, \"required\": [\"query\"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"<image>{image_path}</image>\n{question}"},
        ]

        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round_num = 0
        termination = None
        prediction = ""
        consecutive_errors = 0

        print(f"🚀 Starting rollout generation...")
        print(f"   Image: {image_path}")
        print(f"   Question: {question}")
        print(f"   Max rounds: {MAX_LLM_CALL_PER_RUN}")
        print()

        while num_llm_calls_available > 0:
            round_num += 1
            num_llm_calls_available -= 1

            print(f"--- Round {round_num} ---")

            # Get LLM response
            try:
                response = await self.call_llm(messages)
                content = response["content"]
                print(f"LLM Response (truncated): {content[:200]}...")
            except Exception as e:
                print(f"❌ LLM call failed: {e}")
                prediction = f"LLM error: {e}"
                termination = "error"
                break

            # Check for tool call
            if "<tool_call>" in content and "</tool_call>" in content:
                assistant_msg = {
                    "role": "assistant",
                    "content": content.strip(),
                    "step_error": False,
                }
                messages.append(assistant_msg)

                # Extract tool call
                try:
                    tool_text = content.split("<tool_call>")[1].split("</tool_call>")[0].strip()
                    tool_call = json.loads(tool_text)
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("arguments", {})

                    print(f"🛠️  Executing tool: {tool_name}")
                    print(f"   Args: {json.dumps(tool_args, ensure_ascii=False)[:200]}...")

                    # Execute tool
                    result = await self.execute_tool(tool_name, tool_args, messages)
                    print(f"   Result (truncated): {result[:300]}...")

                    # Check for errors
                    if result.startswith("Error") or "not found" in result.lower():
                        assistant_msg["step_error"] = True
                        consecutive_errors += 1
                    else:
                        consecutive_errors = 0

                except json.JSONDecodeError as e:
                    result = f"[Error] Invalid JSON in tool call: {e}"
                    assistant_msg["step_error"] = True
                    consecutive_errors += 1
                except Exception as e:
                    result = f"[Error] {str(e)}"
                    assistant_msg["step_error"] = True
                    consecutive_errors += 1

                # Add tool response
                tool_response = f"<tool_response>\n{result}\n</tool_response>"
                messages.append({"role": "user", "content": tool_response})

                # Check for too many errors
                if consecutive_errors >= 3:
                    prediction = "Too many consecutive errors"
                    termination = "consecutive_step_errors"
                    break

            # Check for final answer
            elif "<answer>" in content and "</answer>" in content:
                messages.append({
                    "role": "assistant",
                    "content": content.strip(),
                    "step_error": False,
                })
                prediction = content.split("<answer>")[1].split("</answer>")[0].strip()
                termination = "answer"
                consecutive_errors = 0
                print(f"✅ Final answer generated!")
                break

            # Invalid format - no tool call or answer
            else:
                messages.append({
                    "role": "assistant",
                    "content": content.strip(),
                    "step_error": True,
                })
                error_msg = "Error: Response must contain either <tool_call>...</tool_call> or <answer>...</answer> tags. Please try again."
                messages.append({"role": "user", "content": error_msg})
                consecutive_errors += 1
                print(f"⚠️  Invalid format, requesting correction...")

                if consecutive_errors >= 3:
                    prediction = "Too many consecutive format errors"
                    termination = "consecutive_step_errors"
                    break

            print()

            # Check if we've run out of rounds
            if num_llm_calls_available <= 0:
                prediction = f"No answer after {MAX_LLM_CALL_PER_RUN} rounds"
                termination = "max_rounds_reached"
                break

        # Build final result
        time_taken = time.time() - start_time

        result = {
            "question": question,
            "answer": answer,
            "image_path": image_path,
            "messages": messages,
            "prediction": prediction,
            "termination": termination or "unknown",
            "rounds": round_num,
            "time_taken": time_taken,
            "token_usage": {
                "prompt": self.total_prompt_tokens,
                "completion": self.total_completion_tokens,
                "total": self.total_prompt_tokens + self.total_completion_tokens,
            },
        }

        print()
        print("=" * 60)
        print("ROLLOUT COMPLETE")
        print("=" * 60)
        print(f"Rounds: {round_num}")
        print(f"Time: {time_taken:.2f}s")
        print(f"Termination: {termination}")
        print(f"Prediction: {prediction}")
        print(f"Tokens - Prompt: {self.total_prompt_tokens}, Completion: {self.total_completion_tokens}")

        return result

    def save_rollout(self, result: Dict[str, Any], output_dir: str = "./rollouts") -> str:
        """
        Save the rollout to a JSON file.

        Args:
            result: The rollout result dictionary
            output_dir: Directory to save the file

        Returns:
            Path to the saved file
        """
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Handle both raw format (image_path at root) and rllm format (image_path in task)
        image_path = result.get("image_path") or result.get("task", {}).get("image_path", "unknown")
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        filename = f"rollout_{image_name}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return filepath


def format_rollout_for_rllm(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format the rollout result for rLLM training format.

    This converts the rollout into a format compatible with rLLM's Episode structure.

    Args:
        result: The rollout result from RolloutGenerator

    Returns:
        Formatted rollout dictionary
    """
    messages = result.get("messages", [])
    steps = []

    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            # Find the next observation (user message with tool_response)
            observation = ""
            if i + 1 < len(messages) and messages[i + 1].get("role") == "user":
                next_content = messages[i + 1].get("content", "")
                if "<tool_response>" in next_content:
                    observation = next_content

            step = {
                "model_response": msg.get("content", ""),
                "observation": observation,
                "has_error": msg.get("step_error", False),
            }
            steps.append(step)

    formatted = {
        "task": {
            "question": result["question"],
            "answer": result.get("answer"),
            "image_path": result["image_path"],
        },
        "trajectory": {
            "steps": steps,
            "num_steps": len(steps),
        },
        "prediction": result["prediction"],
        "termination": result["termination"],
        "is_correct": False,  # Will be set by reward function
        "metrics": {
            "rounds": result["rounds"],
            "time_taken": result["time_taken"],
            "prompt_tokens": result["token_usage"]["prompt"],
            "completion_tokens": result["token_usage"]["completion"],
        },
    }

    # Evaluate correctness if answer is provided
    if result.get("answer") and result.get("prediction"):
        formatted["is_correct"] = (
            result["answer"].strip().lower() == result["prediction"].strip().lower()
        )

    return formatted


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate rollouts for RL training using agent_v3 tools (vLLM serve version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (requires vLLM serve running on localhost:8000)
    python generate_rollout.py --image_path test_data/location/test5.jpg --question "这是哪?"

    # With ground truth answer
    python generate_rollout.py --image_path test_data/location/test5.jpg --question "这是哪?" --answer "天安门"

    # Custom vLLM serve endpoint
    python generate_rollout.py --image_path test.jpg --question "这是哪?" --base_url http://192.168.1.100:8000/v1

    # Use different model (must match the model loaded in vLLM serve)
    python generate_rollout.py --image_path test.jpg --question "这是哪?" --model qwen3-vl-72b-instruct

Note:
    Make sure vLLM serve is running before executing this script:
        python -m vllm.entrypoints.openai.api_server --model <model_name> --port 8000
        """
    )

    parser.add_argument("--image_path", type=str,default="/gemini-3/space/data/wangcy/Vision-DeepResearch-main/agent_v3/test_data/location/test0.png",
                        help="Path to the input image (same as main2.py)")
    parser.add_argument("--question", type=str, default="这是哪?",
                        help="Question about the image (default: '这是哪?')")
    parser.add_argument("--answer", type=str, default=None,
                        help="Ground truth answer (optional, for evaluation)")
    parser.add_argument("--output_dir", type=str, default="/gemini-3/space/data/baiz/agent_v3/rollouts",
                        help="Directory to save rollout files (default: ./rollouts)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--api_key", type=str, default=VLLM_API_KEY,
                        help="vLLM API key (default: 'EMPTY')")
    parser.add_argument("--base_url", type=str, default=VLLM_BASE_URL,
                        help="vLLM serve base URL (default: http://localhost:8000/v1)")
    parser.add_argument("--format", type=str, choices=["raw", "rllm"], default="rllm",
                        help="Output format: raw (full messages) or rllm (training format)")

    args = parser.parse_args()

    # Create generator
    generator = RolloutGenerator(api_key=args.api_key, model=args.model, base_url=args.base_url)

    # Generate rollout
    result = await generator.run(
        question=args.question,
        image_path=args.image_path,
        answer=args.answer,
    )

    # Format output
    if args.format == "rllm":
        output = format_rollout_for_rllm(result)
    else:
        output = result

    # Save to file
    filepath = generator.save_rollout(output, args.output_dir)
    print(f"\n💾 Rollout saved to: {filepath}")

    # Also print summary
    print("\n" + "=" * 60)
    print("FINAL OUTPUT SUMMARY")
    print("=" * 60)
    print(f"Question: {result['question']}")
    if result.get('answer'):
        print(f"Ground Truth: {result['answer']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Correct: {output.get('is_correct', 'N/A (no ground truth)')}")
    print(f"Total Steps: {result['rounds']}")

    return result


if __name__ == "__main__":
    asyncio.run(main())
