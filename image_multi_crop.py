import json
import math
import os
from io import BytesIO
from math import ceil, floor
from typing import List, Union

import requests
from PIL import Image

from qwen_agent.llm.schema import ContentItem
from qwen_agent.log import logger
from qwen_agent.tools.base import BaseToolWithFileAccess, register_tool
from qwen_agent.utils.utils import extract_images_from_messages
from datetime import datetime


@register_tool('image_multi_crop')
class ImageMultiCrop(BaseToolWithFileAccess):
    description = '图像裁剪工具，根据用户的要求一次性对图片进行裁剪获得多张子图。'
    parameters = {
        'type': 'object',
        'properties': {
            'img_idx': {
                'type': 'number',
                'description': '输入图像的索引'
            },
            'crops': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'bbox_2d': {
                            'type': 'array',
                            'items': {'type': 'number'},
                            'minItems': 4,
                            'maxItems': 4,
                            'description': 'Bounding box as [x1, y1, x2, y2] in pixels or normalized'
                        },
                        'label': {
                            'type': 'string',
                            'description': 'Label for this cropped region'
                        },
                        'sub_idx': {
                            'type': 'number',
                            'description': 'Optional index to distinguish multiple crops (starting from 0)'
                        }
                    },
                    'required': ['bbox_2d']
                },
                'description': 'List of regions to crop from the image'
            }
        },
        'required': ['img_idx', 'crops']
    }

    # --------------------- 工具函数 ---------------------

    def round_by_factor(self, number: int, factor: int) -> int:
        return round(number / factor) * factor

    def ceil_by_factor(self, number: int, factor: int) -> int:
        return math.ceil(number / factor) * factor

    def floor_by_factor(self, number: int, factor: int) -> int:
        return math.floor(number / factor) * factor

    def smart_resize(self, height: int, width: int, factor: int = 32, min_pixels: int = 56 * 56,
                     max_pixels: int = 12845056):
        h_bar = max(factor, self.round_by_factor(height, factor))
        w_bar = max(factor, self.round_by_factor(width, factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = self.floor_by_factor(height / beta, factor)
            w_bar = self.floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = self.ceil_by_factor(height * beta, factor)
            w_bar = self.ceil_by_factor(width * beta, factor)
        return h_bar, w_bar

    def maybe_resize_bbox(self, left, top, right, bottom, img_width, img_height):
        left = max(0, left)
        top = max(0, top)
        right = min(img_width, right)
        bottom = min(img_height, bottom)

        height = bottom - top
        width = right - left
        if height < 32 or width < 32:
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0
            ratio = 32 / min(height, width)
            new_half_height = ceil(height * ratio * 0.5)
            new_half_width = ceil(width * ratio * 0.5)
            new_left = floor(center_x - new_half_width)
            new_right = ceil(center_x + new_half_width)
            new_top = floor(center_y - new_half_height)
            new_bottom = ceil(center_y + new_half_height)

            # 保证不超过图片边界
            new_left = max(0, new_left)
            new_top = max(0, new_top)
            new_right = min(img_width, new_right)
            new_bottom = min(img_height, new_bottom)

            new_height = new_bottom - new_top
            new_width = new_right - new_left

            if new_height > 32 and new_width > 32:
                return [new_left, new_top, new_right, new_bottom]

        return [left, top, right, bottom]

    # --------------------- 主函数 ---------------------

    def call(self, params: Union[str, dict], **kwargs) -> List[ContentItem | dict]:
        params = self._verify_json_format_args(params)
        img_idx = params['img_idx']
        crops = params['crops']

        images = extract_images_from_messages(kwargs.get('messages', []))
        os.makedirs(self.work_dir, exist_ok=True)

        # 打开图片
        try:
            image_arg = images[img_idx]
            if image_arg.startswith('file://'):
                image_arg = image_arg[len('file://'):]
            if image_arg.startswith('http'):
                response = requests.get(image_arg)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            elif os.path.exists(image_arg):
                image = Image.open(image_arg)
            else:
                image = Image.open(os.path.join(self.work_dir, image_arg))
        except Exception as e:
            logger.warning(f'{e}')
            return [ContentItem(text=f'Error: Invalid input image {images}')]

        results = []
        img_width, img_height = image.size

        for i, crop in enumerate(crops):
            bbox = crop['bbox_2d']
            label = crop.get('label', f'crop_{i}')
            sub_idx = crop.get('sub_idx', i)

            # 转像素
            if max(bbox) <= 1.0:  # 小于1就认为是百分比
                abs_x1, abs_y1, abs_x2, abs_y2 = [v * img_width if idx % 2 == 0 else v * img_height for idx, v in
                                                  enumerate(bbox)]
            elif max(bbox) <= 1000:  # 千分比
                abs_x1, abs_y1, abs_x2, abs_y2 = [v / 1000.0 * img_width if idx % 2 == 0 else v / 1000.0 * img_height
                                                  for idx, v in enumerate(bbox)]
            else:  # 已是像素
                abs_x1, abs_y1, abs_x2, abs_y2 = bbox

            validated_bbox = self.maybe_resize_bbox(abs_x1, abs_y1, abs_x2, abs_y2, img_width, img_height)
            left, top, right, bottom = validated_bbox

            cropped_image = image.copy().crop((left, top, right, bottom))

            # Resize
            new_w, new_h = self.smart_resize(right - left, bottom - top, factor=32, min_pixels=256 * 32 * 32)
            cropped_image = cropped_image.resize((new_w, new_h), resample=Image.BICUBIC)

            label_sanitized = label.replace(' ', '_').replace('/', '_')[:30]  # 避免空格/非法字符并限制长度
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # 年月日-时分秒
            output_path = os.path.abspath(os.path.join(
                self.work_dir,
                f'img{img_idx}_sub{sub_idx}_{label_sanitized}_{timestamp}.png'
            ))
            cropped_image.save(output_path)
            # 裁剪后的子图信息
            sub_image_info = {
                "label": label,
                "bbox_2d": bbox,
                "sub_idx": sub_idx,
                "sub_image_path": output_path
            }
            results.append(ContentItem(text=json.dumps(sub_image_info, ensure_ascii=False)))
            results.append(ContentItem(image=output_path))

        return results


def test():
    from qwen_agent.agents import Assistant
    from qwen_agent.utils.output_beautify import typewriter_print, multimodal_typewriter_print

    llm_cfg = {
        'model_type': 'qwenvl_dashscope',
        'model': 'qwen3-vl-plus',
        'api_key': os.getenv('DASHSCOPE_API_KEY', ''),
        'generate_cfg': {
            "top_p": 0.8,
            "top_k": 20,
            "temperature": 0.7,
            "repetition_penalty": 1.0,
            "presence_penalty": 1.5
        }
    }
    #     analysis_prompt = """
    # 你是一个图片裁剪与联网搜索分析工具。你的任务是根据用户的问题，从整张图片中裁剪出最相关的部分，并对其进行分析以回答用户的问题。必要时可以进行联网搜索。请严格遵循以下原则：

    # 1. **主体优先、细节次之**：裁剪时优先选择与用户问题最相关的主体，再考虑细节。
    # 2. **图片中可能拥有多个符合用户需求的对象**： 你需要依次对他们进行处理并多次调用工具
    # 3. **每个对象必须按严格顺序处理**：
    #    - 对每个对象，必须先调用 `image_zoom_in_tool_v2` 进行裁剪。
    #    - **立即**对裁剪后的图片调用 `web_image_search` 进行分析。
    #    - 每个对象完成裁剪和搜索后，才处理下一个对象。
    #    - 不得先对所有对象裁剪完再统一搜索，也不得先搜索原图再裁剪。
    # 4. **裁剪和搜索结果必须被使用**：分析答案必须结合每次裁剪后的 `web_image_search` 输出，不得直接使用模型自身知识回答。

    # """
    analysis_prompt = """
    你是一名智能图像分析助手。
    你的任务是根据用户的要求，从输入图片中识别并裁剪出所有相关的子区域（感兴趣目标）。
    例如，当用户要求“请把照片中每个人都裁剪出来”时，你需要识别照片中的所有人物，并为每个人生成一个裁剪框。
    请一次性完成所有裁剪，不要多次调用工具。

    输出格式：

    {
    "img_idx": 0,
    "crops": [
        {"label": "左边穿白衣服的男人", "bbox_2d": [x1, y1, x2, y2], "sub_idx": 0},
        {"label": "中间戴眼镜的女人", "bbox_2d": [x1, y1, x2, y2], "sub_idx": 1},
        {"label": "右边穿红衣服的小女孩", "bbox_2d": [x1, y1, x2, y2], "sub_idx": 2}
    ]
    }
    """

    tools = [
        'multi_crop',
        # 'web_image_search'
    ]

    agent = Assistant(
        llm=llm_cfg,
        function_list=tools,
        system_message=analysis_prompt
    )

    messages = [
        {'role': 'user',
         'content': [
             {'image': '<your_local_image_path>'},
             # {'text':'这里分别是什么菜？'}
             # {'text':'这是什么电影?'}
             {'text': '这张照片里的演员和饰演的角色分别是谁？'}
         ]}
    ]

    response_plain_text = ''
    for ret_messages in agent.run(messages):
        response_plain_text = multimodal_typewriter_print(ret_messages, response_plain_text)


if __name__ == "__main__":
    test()

