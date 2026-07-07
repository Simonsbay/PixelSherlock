import asyncio
import json
import os
from typing import Union, List, Any

import requests
from PicImageSearch import Network, BaiDu
from bs4 import BeautifulSoup
from openai import AsyncOpenAI

from qwen_agent.llm.schema import ContentItem
from qwen_agent.tools.base import register_tool, BaseTool

from crawl4ai import AsyncWebCrawler, BrowserConfig, DefaultMarkdownGenerator, \
    CrawlerRunConfig, PruningContentFilter

from qwen_agent.utils.utils import extract_images_from_messages


DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', '')
DASHSCOPE_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
QIANFAN_API_KEY = os.getenv('QIANFAN_API_KEY', '')
QIANFAN_BASE_URL = 'https://qianfan.baidubce.com/v2/ai_search/web_search'
IMAGE_SEARCH_TOP_K = 5
TEXT_SEARCH_TOP_K = 5
SUMM_MODEL = 'qwen3-4b'
IMAGE_SEARCH_LLM_SUMM_PROMPT = """
## 系统角色: 图像理解专家

你将接收以下三部分输入内容：
1. **用户问题**: {query}  
   这是用户提出的关于图像的问题，可能涉及图像中的物体、场景、人物、颜色、风格等方面。

2. **图像相关文本**: {text_content}  
   这是与图像相关的网页信息或描述性文本，提供了关于图像的背景信息、上下文，或直接描述了图像中的视觉内容。

3. **百度识图结果**: {baidu_image_title}  
   来自百度图像搜索的结果，通常是一个简短的标题或描述（例如：“Qwen卡通熊”）。这个结果为你提供了图像的标识或分类信息，帮助你进一步理解图像内容。

你的任务是根据用户问题、图像描述文本和百度识图结果，**提炼出与用户问题相关的核心视觉内容**。  
请注意：
- **深入分析文本描述中的视觉信息**：虽然没有直接的图像输入，但你可以从图像相关文本提取出与用户问题有关的核心信息。
- **结合百度识图结果**：当百度识图结果提供的是特定的关键词（如“Qwen卡通熊”），你应当关注这个词汇背后的视觉特征或风格，并将其与用户的查询结合，提供相关的视觉理解。比如，“Qwen卡通熊”可能指的是一种卡通风格的熊，强调其色彩、形状和风格。
- **提供紧密相关的视觉内容摘要**：即使没有直接的图像，仍要确保通过文本和百度识图结果的分析，能够准确提炼出图像的视觉特征，解答用户问题。摘要应当简洁、准确，并且直接与用户的查询需求相吻合。
"""
TEXT_SEARCH_LLM_SUMM_PROMPT = """
## 系统角色: 文本摘要总结专家

你将接收以下三部分输入内容：
1. **用户问题**: {query}  
   用户的问题是基于文本内容的查询，涉及具体的信息需求，可能涉及时间、地点、事件、人物等。

2. **文本内容**: {text_content}  
   这是通过文本搜索引擎返回的网页内容，可能包括文章、新闻、学术论文等形式的文本。

你的任务是从提供的文本中，**提炼出与用户问题最相关的核心信息**。  
请注意：  
- 确保摘要的内容能够回答用户的查询，涵盖问题的所有关键点。
- 强调与用户查询紧密相关的内容，并避免无关信息。
- 关注文本的细节，提炼出有价值的信息，确保摘要清晰、简洁且具有高相关性。
"""


async def anal_web_url(url: str) -> str:
    """crawl4ai解析web网址"""
    try:
        browser_conf = BrowserConfig(
            browser_type="chromium",
            headless=True,
            text_mode=True,
            verbose=False
        )
        run_config = CrawlerRunConfig(
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(
                    threshold=0.9,
                    threshold_type="fixed"
                ),
                options={"ignore_links": True, "ignore_images": True}
            ),
            verbose=False
        )
        async with AsyncWebCrawler(config=browser_conf) as crawler:
            result = await crawler.arun(url, run_config)
            # print(f"解析结果: {result.markdown.fit_markdown}")
            if not result.markdown:
                print(f"web网址: {url} 解析失败，未爬取到网页信息")
                return ""
            if not result.markdown.fit_markdown.strip():
                print(f"web网址: {url} 解析失败，内容过滤后无有效信息")
                return ""
            print(f"web网址: {url} 解析已完成")
        return result.markdown.fit_markdown
    except Exception as e:
        print(f"Error in anal_web_url for {url}: {e}")
        return ""


async def async_anal_urls(urls) -> list[Any]:
    """添加异步任务加速解析"""
    tasks = []
    for url in urls:
        tasks.append(anal_web_url(url))
    try:
        result = await asyncio.gather(*tasks)
    except Exception as e:
        print(f"Error in async_anal_urls: {e}")
        result = []
    return result


async def dashscope_call(messages: list) -> str:
    """调用千问大模型提炼总结信息摘要"""
    try:
        client = AsyncOpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
        )
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


async def async_llm_summ(search_type: str, text_list: list, query: str, baidu_image_title: str = "未找到图像相关信息") -> list[str]:
    tasks = []
    for item in text_list:
        if search_type == "image":
            system_prompt = IMAGE_SEARCH_LLM_SUMM_PROMPT.format(query=query, text_content=item, baidu_image_title=baidu_image_title)
        else:
            system_prompt = TEXT_SEARCH_LLM_SUMM_PROMPT.format(query=query, text_content=item)
        messages = [{'role': 'system', 'content': f'{system_prompt}'}, {'role': 'user', 'content': f'{query}'}]
        tasks.append(dashscope_call(messages))
    try:
        summ_res = await asyncio.gather(*tasks)
        summ_res = list(summ_res)
        print(f"百度{'图搜' if search_type == 'image' else '文搜'} top_k={len(summ_res)} 通义千问提炼总结:\n{summ_res}")
        return summ_res
    except Exception as e:
        print(f"Error in async_llm_summ: {e}")
        return []


@register_tool('image_text_web_search')
class ImageTextWebSearch(BaseTool):
    name = 'image_text_web_search'
    description = '从互联网上搜索相关信息，支持视觉搜索和文本搜索两种方式。视觉搜索用于理解图像内容，文本搜索用于基于查询问题提取网页信息。'
    parameters = {
        'type': 'object',
        'properties': {
            'img_idx': {
                'type': 'number',
                'description': 'The index of the image (starting from 0)'
            },
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
            'search_type': {
                'type': 'string',
                'description': (
                    '指定搜索类型：\n'
                    '- **image**: 侧重于图像理解的搜索，主要用于提取与图像相关的视觉信息或通过图像相关内容来解答用户的查询。\n'
                    '- **web**: 侧重于基于用户查询问题的文本搜索，主要通过解析网页内容来获取相关的答案。\n'
                    '选择 **image** 时，搜索的焦点将放在图像内容的分析和理解上；选择 **web** 时，系统将专注于基于查询内容从网页中获取并提炼出相关信息。'
                ),
                'enum': ['image', 'web']
            }
        },
        'required': ['img_idx', 'query', 'search_type']
    }

    def call(self, params: Union[str, dict], **kwargs) -> Union[str, list, List[ContentItem]]:
        params = self._verify_json_format_args(params)
        query = params['query']
        search_type = params['search_type']
        image_id = int(params['img_idx'])
        images = extract_images_from_messages(kwargs.get('messages', []))
        if not images:
            return 'Error: no images found in the messages.'
        if image_id >= len(images):
            image_id = len(images) - 1
        image_url = images[image_id]
        search_results = asyncio.run(self.search(image_url, query, search_type))
        if not search_results:
            return 'The ImageTextWebSearch tool could not find relevant reference information'
        return search_results

    @staticmethod
    async def search(image_url: str, query: str, search_type: str) -> list[Any]:
        try:
            # 图搜+文搜 双路并行搜索
            # baidu_image_res, baidu_text_res = await asyncio.gather(
            #     baidu_image_search(image_url),
            #     baidu_text_search(query)
            # )
            # urls = baidu_image_res.get("urls", [])
            # baidu_image_title = baidu_image_res.get("baidu_image_title", "")
            #
            # anal_res = await async_anal_urls(urls)
            # image_search_res, text_search_res = await asyncio.gather(
            #     async_llm_summ('图搜', anal_res, query, baidu_image_title),
            #     async_llm_summ('文搜', baidu_text_res, query, baidu_image_title)
            # )
            #
            # baidu_search_res = image_search_res + text_search_res
            # return baidu_search_res

            if search_type == 'image':
                baidu_image_res = await baidu_image_search(image_url, top_k=IMAGE_SEARCH_TOP_K)
                urls = baidu_image_res.get("urls", [])
                baidu_image_title = baidu_image_res.get("baidu_image_title", "")
                anal_res = await async_anal_urls(urls)
                image_search_res = await async_llm_summ(search_type, anal_res, query, baidu_image_title)
                return image_search_res

            baidu_text_res = baidu_text_search(query, top_k=TEXT_SEARCH_TOP_K)
            text_search_res = await async_llm_summ(search_type, baidu_text_res, query)
            return text_search_res

        except Exception as e:
            print(f"Error in search: {e}")
            return []


def extra_baidu_image_result(url) -> str:
    try:
        # 1. 发送请求获取网页内容
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        html_content = response.text

        # 2. 定位包含 window.cardData 的 script 标签
        soup = BeautifulSoup(html_content, 'html.parser')
        script_tag = None
        for script in soup.find_all('script'):
            if script.string and "window.cardData" in script.string:
                script_tag = script
                break

        if not script_tag:
            print("❌ 未找到包含 window.cardData 的 script 标签")
            # 调试：打印所有 script 标签的部分内容（帮助确认是否有目标数据）
            print("\n调试：所有 script 标签片段（前100字符）：")
            for i, s in enumerate(soup.find_all('script')):
                if s.string:
                    print(f"script {i}: {s.string[:100]}...")
            return ""

        # 3. 提取 script 内容（简化处理，忽略换行/空格差异）
        script_content = script_tag.string.replace('\n', ' ').replace('\r', ' ')  # 统一换行符为空格
        script_content = ' '.join(script_content.split())  # 合并连续空格

        # 4. 灵活定位 window.cardData 的范围（核心优化）
        # 起始：找到 "window.cardData = ["
        start_marker = "window.cardData =["
        start_idx = script_content.find(start_marker)
        if start_idx == -1:
            print("❌ 未找到 window.cardData 的起始标记")
            return ""

        # 结束：从起始位置后找最近的 "];", 因为数组一定以 "];"" 结束
        end_idx = script_content.find("];", start_idx)
        if end_idx == -1:
            print("❌ 未找到 window.cardData 的结束标记（];）")
            # 调试：打印找到的 script 内容片段（帮助确认格式）
            print(f"\n调试：找到的 script 内容片段（start_idx附近）：")
            print(script_content[max(0, start_idx - 100): start_idx + 200] + "...")
            return ""

        # 提取完整 JSON 字符串（包含 []）
        card_data_str = script_content[start_idx + len(start_marker) - 1: end_idx + 1]

        # 5. 解析 JSON 并提取目标值
        card_data = json.loads(card_data_str)
        sub_title = card_data[0]["tplData"]["pano"]["list"][0]["subTitle"]
        print(f"\n百度识图: {sub_title}")
        return str(sub_title)
    except requests.exceptions.RequestException as e:
        print(f"HTTP request error: {e}")
        return ""
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return ""
    except KeyError as e:
        print(f"Data extraction error: {e}")
        return ""
    except Exception as e:
        print(f"Error in extra_baidu_image_result: {e}")
        return ""


async def baidu_image_search(image_url: str, top_k: int = 2) -> dict:
    """百度图搜"""
    try:
        async with Network() as client:
            baidu = BaiDu(client=client)
            baidu_resp = await baidu.search(url=image_url)
        urls = []
        for item in baidu_resp.raw:
            if item.url and hasattr(item, "url"):
                urls.append(item.url)
            if len(urls) >= top_k:
                break
        baidu_image_title = extra_baidu_image_result(baidu_resp.url)
        res = {
            "urls": urls,
            "baidu_image_title": baidu_image_title
        }
        return res
    except requests.exceptions.RequestException as e:
        print(f"HTTP request error: {e}")
        return {}

    except Exception as e:
        print(f"Error in baidu_image_search: {e}")
        return {}


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
) -> list:
    """百度文搜"""
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
