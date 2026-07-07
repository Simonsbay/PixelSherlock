import requests
from PIL import Image
from io import BytesIO
import cv2
import glob
import os
from datetime import datetime
import argparse
from functools import partial
from IPython.display import display, Image as IPImage
import torch.multiprocessing as mp
from multiprocessing import Queue
import time
import warnings
warnings.filterwarnings('ignore')

def process_image(url):
    img = cv2.imread(url)
    url = url.replace(".webp",".jpg")
    url = url.replace(".png",".jpg")
    cv2.imwrite(url,img)
    return url

def get_image_files(folder_path, extensions=('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
    """获取文件夹中的所有图片文件"""
    image_files = []
    folder_path = os.path.abspath(folder_path)
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))
    return sorted(list(set(image_files)))


def worker_process(gpu_id, image_files, prompt, llm_config, output_dir_base, result_queue):
    """
    工作进程：在指定GPU上运行SAM3推理

    Args:
        gpu_id: GPU设备ID
        image_files: 分配给该GPU的图片列表
        prompt: 推理提示词
        llm_config: LLM配置
        output_dir_base: 输出目录基础路径
        result_queue: 结果队列，用于返回处理结果
    """
    # 设置当前进程只使用指定的GPU - 必须在任何cuda操作之前设置
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 导入必须在设置CUDA_VISIBLE_DEVICES之后
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.inference_mode().__enter__()

    # 显式设置默认cuda设备
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # 在当前进程中就是唯一的GPU

    SAM3_ROOT = os.path.dirname(os.getcwd())
    os.chdir(SAM3_ROOT)

    # 导入SAM3相关模块
    import sam3
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.agent.client_llm import send_generate_request as send_generate_request_orig
    from sam3.agent.client_sam3 import call_sam_service as call_sam_service_orig
    from sam3.agent.inference import run_single_image_inference

    # 初始化SAM3模型
    sam3_root = os.path.dirname(sam3.__file__)
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"

    # 使用cuda:0，因为在设置了CUDA_VISIBLE_DEVICES后，每个进程只能看到自己的GPU作为cuda:0
    device = "cuda:0"

    # 构建模型并确保它在GPU上
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        load_from_HF=False,
        checkpoint_path="/baizhen/workspace/sam3.pt",
        device=device
    )

    # 显式将模型移动到GPU并确保所有参数都在GPU上
    model = model.to(device)

    # 强制所有模型参数和buffer到GPU（解决权重在CPU上的问题）
    for param in model.parameters():
        if not param.is_cuda:
            param.data = param.data.to(device)
    for buffer in model.buffers():
        if not buffer.is_cuda:
            buffer.data = buffer.data.to(device)

    # 确保模型处于eval模式
    model.eval()

    # 检查第一个参数的设备位置
    first_param_device = next(model.parameters()).device
    print(f"[GPU {gpu_id}] 模型已加载到设备: {first_param_device}")

    processor = Sam3Processor(model, confidence_threshold=0.5)

    # 设置LLM API
    LLM_CONFIGS = {
        "Qwen3-VL-30B-A3B-Instruct-FP8": {
            "provider": "vllm",
            "model": "/workspace/models/",
        },
    }

    model_name = "Qwen3-VL-30B-A3B-Instruct-FP8"
    llm_config_local = LLM_CONFIGS[model_name]
    llm_config_local["api_key"] = "DUMMY_API_KEY"
    llm_config_local["name"] = model_name

    if llm_config_local["provider"] == "vllm":
        LLM_SERVER_URL = "http://0.0.0.0:8000/v1"
    else:
        LLM_SERVER_URL = llm_config["base_url"]

    # 创建偏函数
    send_generate_request = partial(
        send_generate_request_orig,
        server_url=LLM_SERVER_URL,
        model=llm_config_local["model"],
        api_key=llm_config_local["api_key"]
    )
    call_sam_service = partial(call_sam_service_orig, sam3_processor=processor)

    # 处理图片
    results = []
    current_time = datetime.now().strftime("%m%d_%H%M%S")
    output_dir = os.path.join(output_dir_base, f"gpu_{gpu_id}_{current_time}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[GPU {gpu_id}] 开始处理 {len(image_files)} 张图片")

    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[GPU {gpu_id}] 处理 [{idx}/{len(image_files)}]: {image_path}")

        img_name = os.path.splitext(os.path.basename(image_path))[0]
        img_output_dir = os.path.join(output_dir, img_name)

        try:
            processed_image = process_image(image_path)
            output_image_path = run_single_image_inference(
                processed_image, prompt, llm_config_local, send_generate_request, call_sam_service,
                debug=True, output_dir=img_output_dir
            )

            if output_image_path is not None:
                print(f"[GPU {gpu_id}] ✅ [{idx}/{len(image_files)}] 成功: {output_image_path}")
                results.append({"status": "success", "path": output_image_path, "gpu": gpu_id})
            else:
                print(f"[GPU {gpu_id}] ❌ [{idx}/{len(image_files)}] 失败: {image_path}")
                results.append({"status": "failed", "path": image_path, "gpu": gpu_id})
        except Exception as e:
            print(f"[GPU {gpu_id}] ❌ [{idx}/{len(image_files)}] 错误: {image_path} - {str(e)}")
            results.append({"status": "error", "path": image_path, "gpu": gpu_id, "error": str(e)})

    print(f"\n[GPU {gpu_id}] 处理完成! 结果保存至: {output_dir}")

    # 将结果放入队列
    result_queue.put({"gpu_id": gpu_id, "results": results, "output_dir": output_dir})


def main():
    parser = argparse.ArgumentParser(description='SAM3 Agent Inference with Multi-GPU Parallel Processing')
    parser.add_argument('--image', type=str, default='',
                        help='Path to the input image (for single image inference)')
    parser.add_argument('--image-folder', type=str, default="/baizhen/workspace/testdata/fight_all_images",
                        help='Path to folder containing images (for batch inference)')
    parser.add_argument('--prompt', type=str, default='People who are arguing or fighting',
                        help='Text prompt for the inference')
    parser.add_argument('--gpus', type=str, default='0,2,3',
                        help='Comma-separated list of GPU IDs to use (default: 0,2,3)')
    args = parser.parse_args()

    prompt = args.prompt

    # 解析GPU列表
    gpu_ids = [int(g.strip()) for g in args.gpus.split(',')]
    print(f"使用GPU: {gpu_ids}")

    # 添加时间戳到输出目录
    current_time = datetime.now().strftime("%m%d_%H%M%S")
    output_dir_base = f"/baizhen/workspace/new-agent/agent_res_fight/{current_time}"
    os.makedirs(output_dir_base, exist_ok=True)

    # 判断是单图模式还是批量文件夹模式
    if args.image_folder:
        # 批量处理文件夹中的图片
        image_folder = os.path.abspath(args.image_folder)
        all_image_files = get_image_files(image_folder)

        if not all_image_files:
            print(f"错误: 在文件夹 {image_folder} 中没有找到图片文件")
            return

        print(f"\n=== 批量处理 {len(all_image_files)} 张图片 ===")
        print(f"文件夹: {image_folder}")
        print(f"提示词: {prompt}")
        print(f"输出目录: {output_dir_base}")
        print(f"使用GPU: {gpu_ids}")

        # 将图片平均分配给各个GPU
        num_gpus = len(gpu_ids)
        image_splits = [[] for _ in range(num_gpus)]

        for i, image_path in enumerate(all_image_files):
            gpu_idx = i % num_gpus
            image_splits[gpu_idx].append(image_path)

        # 显示分配情况
        for i, (gpu_id, images) in enumerate(zip(gpu_ids, image_splits)):
            print(f"  GPU {gpu_id}: {len(images)} 张图片")

        # 创建结果队列
        result_queue = Queue()

        # 创建并启动工作进程
        processes = []
        start_time = time.time()

        for gpu_id, image_list in zip(gpu_ids, image_splits):
            if not image_list:
                continue

            p = mp.Process(
                target=worker_process,
                args=(gpu_id, image_list, prompt, {}, output_dir_base, result_queue)
            )
            p.start()
            processes.append(p)

        # 等待所有进程完成
        all_results = []
        for p in processes:
            p.join()

        # 收集结果
        while not result_queue.empty():
            all_results.append(result_queue.get())

        end_time = time.time()
        total_time = end_time - start_time

        # 汇总结果
        print(f"\n{'='*60}")
        print(f"批量处理完成!")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"结果保存至: {output_dir_base}")

        total_success = 0
        total_failed = 0
        for res in all_results:
            gpu_results = res["results"]
            success_count = sum(1 for r in gpu_results if r["status"] == "success")
            failed_count = len(gpu_results) - success_count
            total_success += success_count
            total_failed += failed_count
            print(f"  GPU {res['gpu_id']}: 成功 {success_count}, 失败 {failed_count}")

        print(f"总计: 成功 {total_success}, 失败 {total_failed}")
        print(f"{'='*60}")

    else:
        # 单图处理模式 - 使用第一个GPU
        image = args.image
        if not image:
            print("错误: 请提供 --image 或 --image-folder 参数")
            return

        image = os.path.abspath(image)

        print("\n=== 单图处理模式 ===")
        print(f"输入图片: {image}")
        print(f"提示词: {prompt}")
        print(f"使用GPU: {gpu_ids[0]}")

        # 创建结果队列
        result_queue = Queue()

        # 启动单进程
        p = mp.Process(
            target=worker_process,
            args=(gpu_ids[0], [image], prompt, {}, output_dir_base, result_queue)
        )
        p.start()
        p.join()

        # 获取结果
        if not result_queue.empty():
            result = result_queue.get()
            print(f"\n处理完成! 结果保存至: {result['output_dir']}")


if __name__ == "__main__":
    # 多进程必须使用这个保护
    # 对于CUDA多进程，必须使用spawn模式
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 已经设置过了
    main()
