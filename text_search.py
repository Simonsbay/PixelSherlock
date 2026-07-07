import asyncio
import json
import os
from typing import Union, List, Any

import requests
from openai import AsyncOpenAI

from qwen_agent.llm.schema import ContentItem
from qwen_agent.tools.base import register_tool, BaseTool

DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', '')
DASHSCOPE_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
QIANFAN_API_KEY = os.getenv('QIANFAN_API_KEY', '')
QIANFAN_BASE_URL = 'https://qianfan.baidubce.com/v2/ai_search/web_search'
TEXT_SEARCH_TOP_K = 5
SUMM_MODEL = 'qwen3-4b'
TEXT_SEARCH_LLM_SUMM_PROMPT = """
## 系统角色: 文本摘要总结专家

### 输入内容
你将接收以下两部分输入内容
1. **用户问题**: {query}  
   - 说明：用户的问题是基于文本内容的查询，涉及具体的信息需求，可能涉及时间、地点、事件、人物等。

2. **文本内容**: {text_content}  
   - 说明：这是通过文本搜索引擎返回的网页内容，可能包括文章、新闻、学术论文等形式的文本。

### 核心任务
从提供的文本中，**提炼出与用户问题最相关的核心信息**。  

### 严格要求
1. 保证摘要内容在100字以内。  
2. 确保摘要的内容能够回答用户的查询，涵盖问题的所有关键点。
3. 强调与用户查询紧密相关的内容，并避免无关信息。
4. 关注文本的细节，提炼出有价值的信息，确保摘要清晰、简洁且具有高相关性。
"""


@register_tool('web_text_search')
class WebSearch(BaseTool):
    name = 'web_text_search'
    description = '网络搜索工具。当通过图像工具识别出视觉主体以及相关信息后，需要使用此工具联网搜索补充信息'
    parameters = {
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
                'description': (
                    '基于对话上下文生成的具体查询问题，需满足以下要求：\n'
                    '1. 完整覆盖上下文待解答的核心需求，不遗漏关键信息；\n'
                    '2. 表述清晰、简洁，符合搜索引擎的查询习惯；\n'
                    '3. 避免模糊、宽泛的表述，尽可能包含关键限定词（如时间、地点、主体、具体维度等）；\n'
                    '4. 以自然语言问题形式呈现（而非关键词堆砌），例如："2025年人工智能在医疗行业的应用前景"、"多智能体协作在自动驾驶中的优势"。'
                )
            },
        },
        'required': ['query']
    }

    def call(self, params: Union[str, dict], **kwargs) -> Union[str, list, List[ContentItem]]:
        params = self._verify_json_format_args(params)
        query = params['query']
        search_results = asyncio.run(self.search(query))
        if not search_results:
            return 'The WebSearch tool could not find relevant reference information'
        return search_results

    @staticmethod
    async def search(query: str) -> list[str]:
        try:
            baidu_text_res = baidu_text_search(query, top_k=TEXT_SEARCH_TOP_K)
            text_search_res = await async_llm_summ(baidu_text_res, query)
            return text_search_res

        except Exception as e:
            print(f"Error in search: {e}")
            return []


async def dashscope_call(messages: list) -> str:
    """调用千问大模型提炼总结信息摘要"""
    try:
        async with AsyncOpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL) as client:
            completion = await client.chat.completions.create(
                model=SUMM_MODEL,
                messages=messages,
                stream=False,
                extra_body={"enable_thinking": False}
            )
            res = completion.choices[0].message.content
            # print(f"模型提炼结果:\n{res}")
            return res
    except Exception as e:
        print(f"Error in dashscope_call: {e}")
        return ""


async def async_llm_summ(text_list: list, query: str) -> list[str]:
    tasks = []
    for item in text_list:
        system_prompt = TEXT_SEARCH_LLM_SUMM_PROMPT.format(query=query, text_content=item)
        messages = [{'role': 'system', 'content': f'{system_prompt}'}, {'role': 'user', 'content': f'{query}'}]
        tasks.append(dashscope_call(messages))
    try:
        summ_res = await asyncio.gather(*tasks)
        summ_res = list(summ_res)
        print(f"百度文搜---通义千问提炼总结:\n{summ_res}")
        return summ_res
    except Exception as e:
        print(f"Error in async_llm_summ: {e}")
        return []


def build_payload(
        query,
        search_source,
        top_k,
        search_recency_filter,
        resource_type
) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": query},
        ],
        "search_source": search_source,
        "resource_type_filter": [{"type": resource_type, "top_k": top_k}],
        "search_recency_filter": search_recency_filter,
    }


def baidu_text_search(
        query: str,
        top_k: int = 2,
        search_source: str = 'baidu_search_v2',
        search_recency_filter: str = 'year',
        resource_type: str = 'web'
) -> list[str]:
    """百度文搜（千帆API）"""
    url = QIANFAN_BASE_URL
    headers = {
        'X-Appbuilder-Authorization': f'Bearer {QIANFAN_API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = build_payload(query, search_source, top_k, search_recency_filter, resource_type)
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        # print(response.json())
        references = response.json().get("references")
        if not references:
            raise ValueError("No references found in the response.")
        baidu_text_res = []
        for reference in references:
            baidu_text_res.append(reference.get("content"))
        # print(f"百度文搜结果:\n{baidu_text_res}")
        return baidu_text_res
    except requests.exceptions.RequestException as e:
        print(f"Http request error:: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return []
    except ValueError as e:
        print(f"Data error: {e}")
        return []
    except Exception as e:
        print(f"Error in baidu_text_search: {e}")
        return []


if __name__ == '__main__':
    baidu_text_search('通义千问')
