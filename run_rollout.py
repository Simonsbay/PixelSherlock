#!/usr/bin/env python3
"""
Agent V3 Rollout Runner - Executable Script

This script generates complete rollouts for RL training.
It accepts the same input format as main2.py.

Usage:
    python run_rollout.py --image test_data/location/test5.jpg --question "这是哪?"
    python run_rollout.py --image test_data/location/test5.jpg --question "这是哪?" --answer "天安门"

Output:
    - Complete rollout JSON saved to ./rollouts/
    - Summary printed to console
    - Compatible with rLLM training format
"""

import asyncio
import argparse
import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/gemini-3/space/data/wangcy/Vision-DeepResearch-main/rllm/vision_deepresearch_async_workflow')

from agent_v3_workflow import AgentV3Workflow


async def main():
    parser = argparse.ArgumentParser(
        description='Generate rollout for RL training (same input as main2.py)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    # Same input as main2.py
    python run_rollout.py --image test_data/location/test5.jpg --question "这是哪?"

    # With ground truth for evaluation
    python run_rollout.py --image test_data/location/test5.jpg --question "这是哪?" --answer "天安门"

    # Custom output directory
    python run_rollout.py -i test.jpg -q "这是哪?" -o ./my_rollouts
        '''
    )

    parser.add_argument('-i', '--image', '--image_path', dest='image_path',
                        required=True,
                        help='Path to the input image (same as main2.py)')

    parser.add_argument('-q', '--question',
                        default='这是哪?',
                        help='Question about the image (default: "这是哪?")')

    parser.add_argument('-a', '--answer',
                        default=None,
                        help='Ground truth answer (optional, for evaluation)')

    parser.add_argument('-o', '--output_dir',
                        default='./rollouts',
                        help='Output directory (default: ./rollouts)')

    parser.add_argument('--save_messages',
                        action='store_true',
                        help='Save full message history (for debugging)')

    args = parser.parse_args()

    # Verify image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found: {args.image_path}")
        sys.exit(1)

    # Get absolute path
    image_path = os.path.abspath(args.image_path)

    print("=" * 70)
    print("Agent V3 Rollout Generator")
    print("=" * 70)
    print(f"Image: {image_path}")
    print(f"Question: {args.question}")
    if args.answer:
        print(f"Ground Truth: {args.answer}")
    print("-" * 70)

    # Create workflow
    workflow = AgentV3Workflow()

    # Create task
    task = {
        "question": args.question,
        "images": [image_path],
        "answer": args.answer,
    }

    # Generate rollout
    episode = await workflow.run(task, uid=f"rollout_{int(asyncio.get_event_loop().time())}")

    # Prepare output
    output = {
        "task": episode.task,
        "prediction": episode.info.get("prediction"),
        "answer": episode.info.get("answer"),
        "is_correct": episode.is_correct,
        "termination": str(episode.termination_reason),
        "rounds": episode.info.get("rounds"),
        "time_taken": episode.info.get("time_taken"),
        "token_usage": episode.info.get("token_usage"),
    }

    if args.save_messages:
        output["trajectory"] = {
            "steps": [
                {
                    "model_response": step.model_response,
                    "observation": step.observation,
                    "has_error": step.info.get("step_error", False),
                }
                for step in episode.trajectories[0].steps
            ] if episode.trajectories else []
        }

    # Save output
    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = str(int(asyncio.get_event_loop().time()))
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    filename = f"rollout_{image_name}_{timestamp}.json"
    filepath = os.path.join(args.output_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("ROLLOUT SUMMARY")
    print("=" * 70)
    print(f"Prediction: {output['prediction']}")
    print(f"Answer: {output['answer']}")
    print(f"Correct: {output['is_correct']}")
    print(f"Rounds: {output['rounds']}")
    print(f"Time: {output['time_taken']:.2f}s")
    print(f"Tokens: {output['token_usage']}")
    print("-" * 70)
    print(f"Saved to: {filepath}")
    print("=" * 70)


if __name__ == '__main__':
    asyncio.run(main())
