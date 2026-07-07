from typing import Union, List

from qwen_agent.llm.schema import ContentItem
from qwen_agent.tools.base import register_tool, BaseTool


@register_tool('image_analyse')
class ImageAnalyse(BaseTool):
    name = 'image_analyse'
    description = '图像识别工具，识别图像中的视觉主体并获取与其相关的信息。'
    parameters = {
        'type': 'object',
        'properties': {
            'sub_image_info_list': {
                'type': 'array',
                'description': '''筛选出来的子图列表，示例如下：
                    [{
                        "selected_subject": "匹配主体的label",
                        "selected_sub_idx": 子图编号,
                        "reason": "说明选择原因",
                        "sub_image_path": "子图路径"
                    }]
                ''',
                'items': {
                    'type': 'object',
                    'properties': {
                        'selected_subject': {
                            'type': 'string',
                            'description': '匹配主体的label'
                        },
                        'selected_sub_idx': {
                            'type': 'integer',
                            'description': '子图编号'
                        },
                        'reason': {
                            'type': 'string',
                            'description': '说明选择原因'
                        },
                        'sub_image_path': {
                            'type': 'string',
                            'description': '子图路径'
                        }
                    },
                    'required': ['selected_subject', 'selected_sub_idx', 'reason', 'sub_image_path']
                }
            },
        },
        'required': ['sub_image_info_list']
    }

    def call(self, params: Union[str, dict], **kwargs) -> Union[str, list, List[ContentItem]]:
        return [ContentItem(text="请基于自身知识回答用户问题")]
