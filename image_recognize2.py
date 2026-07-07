import asyncio
import json
import os
from typing import Union, List
import requests
from PicImageSearch import Network, BaiDu
from bs4 import BeautifulSoup
from openai import AsyncOpenAI
from qwen_agent.llm.schema import ContentItem
from qwen_agent.tools.base import register_tool, BaseTool
from crawl4ai import AsyncWebCrawler, BrowserConfig, DefaultMarkdownGenerator, CrawlerRunConfig, PruningContentFilter

DASHSCOPE_API_KEY = os.environ.get('DASHSCOPE_API_KEY', '')
DASHSCOPE_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
IMAGE_SEARCH_TOP_K = 5
SUMM_MODEL = 'qwen3-4b'
BLACKLIST_DOMAINS = ['h5.weishi.qq.com', 'www.ixigua.com', 'www.xiaohongshu.com', 'www.douyin.com', 'iqiyi.com',
                     'youku.com', 'videos', 'video','mafengwo']
IMAGE_SEARCH_LLM_SUMM_PROMPT = """
## 系统角色: 视觉信息总结专家

### 输入内容
你将接收以下三部分输入内容：
1. **用户问题**: {query}  
   - 说明：这是用户提出的关于图像的问题，可能涉及图像中的物体、场景、人物、颜色、风格等方面。

2. **百度识图结果**: {baidu_image_title}  
   - 说明：来自百度图像搜索的结果，通常是一个简短的标题或描述。这个结果为你提供了图像的标识或分类信息，帮助你进一步理解图像内容。

2. **图像相关文本**: {text_content}  
   - 说明：这是与图像相关的网页信息或描述性文本，提供了关于图像的背景信息、上下文，或直接描述了图像中的视觉内容。

### 核心任务
根据用户问题、图像描述文本和百度识图结果，**提炼出与用户问题相关的核心内容**。  

### 注意
1. **深入分析文本描述中的视觉信息**：虽然没有直接的图像输入，但你可以从图像相关文本提取出与用户问题有关的核心信息。
2. **结合百度识图结果**：当百度识图结果提供的是特定的关键词，你应当关注这个词汇背后的视觉特征或风格，并将其与用户的查询结合，提供相关的视觉理解。
3. **提供紧密相关的视觉内容摘要**：即使没有直接的图像，仍要确保通过文本和百度识图结果的分析，能够准确提炼出图像的视觉特征，解答用户问题。摘要应当简洁、准确，并且直接与用户的查询需求相吻合。

### 严格要求
1. 保证摘要内容在90字以内。  
2. 确保摘要的内容不遗漏所有关键点。
"""


@register_tool('image_recognize')
class ImageRecognize(BaseTool):
    name = 'image_recognize'
    description = 'Image search engine, input the image and search for similar images with image information.'
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
                            'description': '子图编号（从0开始）'
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

    def call(self, params: Union[str, dict], **kwargs) -> Union[str, list, dict, List[ContentItem]]:
        params = self._verify_json_format_args(params)
        sub_image_info_list = params['sub_image_info_list']
        user_question = extract_question_from_messages(kwargs.get('messages', []))
        search_results = asyncio.run(self.search(sub_image_info_list, user_question))
        # print("图搜工具返回相关信息----------------", search_results)
        if not search_results:
            return 'The ImageRecognize tool could not find relevant reference information'
        return search_results

    @staticmethod
    async def search(image_info_list: list, query: str) -> dict[str, list]:
        try:
            all_url = {}
            baidu_image_titles = []
            for idx, image_info in enumerate(image_info_list):
                urls, baidu_image_title = await baidu_image_search(image_info, IMAGE_SEARCH_TOP_K, idx)
                baidu_image_titles.append(baidu_image_title)
                all_url[idx] = urls
            anal_res = await async_anal_urls(all_url)
            image_search_res = await async_llm_summ(anal_res, query, image_info_list, baidu_image_titles,all_url)
            summ_res = []
            for key in image_search_res:
                content = image_search_res[key]
                summ_res.extend(content)
            reformat_res = {
                "summ_res":summ_res,
                "urls":all_url
                }
            return reformat_res
        except Exception as e:
            print(f"Error in search: {e}")
            return {}


async def baidu_image_search(image_info: dict, top_k: int, idx: int) -> tuple[list, str]:
    """百度识图"""
    try:
        async with Network() as client:
            baidu = BaiDu(client=client)
            baidu_resp = await baidu.search(file=image_info.get('sub_image_path'))
        urls = []
        for item in baidu_resp.raw:
            if item.url and hasattr(item, "url"):
                if any(domain in item.url for domain in BLACKLIST_DOMAINS):
                    continue
                urls.append(item.url)
            if len(urls) >= top_k:
                break
        baidu_image_title = extra_baidu_image_result(baidu_resp.url)
        print(f"百度识图 | 目标图像{idx}: {image_info['sub_image_path']} | 识别结果: {baidu_image_title}")
        return urls, baidu_image_title
    except requests.exceptions.RequestException as e:
        print(f"HTTP request error: {e}")
        return [], "未搜索到图像title"
    except Exception as e:
        print(f"百度识图失败（idx={idx}）: {e}")
        return [], "未搜索到图像title"


async def async_llm_summ(anal_res: dict, query: str, image_info_list: list, baidu_image_titles: list,all_url: list) -> dict:
    """异步调用qwen3-4b，针对每一个网页urls解析结果，提炼总结文本摘要"""
    tasks = []
    for key, url_anal_res_list in anal_res.items():
        for url_anal_res in url_anal_res_list:
            system_prompt = IMAGE_SEARCH_LLM_SUMM_PROMPT.format(query=query, text_content=url_anal_res,
                                                                baidu_image_title=baidu_image_titles[key])
            messages = [{'role': 'system', 'content': f'{system_prompt}'}, {'role': 'user', 'content': f'{query}'}]
            tasks.append(dashscope_call(messages))
    try:
        results = await asyncio.gather(*tasks)
    except Exception as e:
        print(f"异步任务（qwen总结）引发异常: {e}")
        return {}
    result_dict = {}
    idx = 0
    for key, url_list in anal_res.items():
        result_dict[key] = []
        for _ in url_list:
            result_dict[key].append(results[idx])
            idx += 1
    format_res = {}
    for key, value in result_dict.items():
        format_res[image_info_list[key].get('selected_subject')] = value
    print("!!!!!!",format_res)
    return format_res


async def async_anal_urls(urls: dict[int, list]) -> dict:
    """异步解析所有子图的相关网页urls"""
    result_dict = {}
    for key in range(len(urls)):
        result_dict[key] = []
    crawl_urls = [url for url_list in urls.values() for url in url_list]
    print(f"开始对所有urls并行爬取与解析: {len(crawl_urls)}个\n{crawl_urls}")
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
                    threshold=0.7,
                    threshold_type="fixed"
                ),
                options={"ignore_links": True, "ignore_images": True}
            ),
            verbose=False
        )
        async with AsyncWebCrawler(config=browser_conf) as crawler:
            results = await crawler.arun_many(
                urls=crawl_urls,
                config=run_config,
            )
            for res in results:
                if not res.markdown:
                    print(f"web网址: {res.url} 解析失败，未爬取到网页信息")
                    continue
                if not res.markdown.fit_markdown.strip():
                    print(f"web网址: {res.url} 解析失败，内容过滤后无有效信息")
                    continue
                for sub_img_idx, url_list in urls.items():
                    if res.url in url_list:
                        result_dict[sub_img_idx].append(res.markdown.fit_markdown)
                        print(f"web网址: {res.url} 解析成功")
            return result_dict

    except Exception as e:
        print(f"Error in async_anal_urls for {urls}: {e}")
        return {}


async def dashscope_call(messages: list) -> str:
    """调用阿里云通义千问大模型（DashScope）"""
    try:
        async with AsyncOpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL) as client:
            completion = await client.chat.completions.create(
                model=SUMM_MODEL,
                messages=messages,
                stream=False,
                extra_body={"enable_thinking": False}
            )
            res = completion.choices[0].message.content
            return res
    except Exception as e:
        print(f"Error in dashscope_call: {e}")
        return ""


def extract_question_from_messages(message):
    role_user = []
    for m in message:
        if m["role"] == "user":
            role_user.append(m)
    # 返回第一次用户提问
    # 临时方案，在多轮对话时会存在问题！！！
    return role_user[0]["content"][1]["text"]


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
        if card_data[0]["tplData"]["pano"] is None:
            print(f'百度识图未提取到有效信息: card_data[0]["tplData"]["pano"] is None')
            return "百度识图未提取到有效信息"
        sub_title = card_data[0]["tplData"]["pano"]["list"][0]["subTitle"]
        # print(f"\n百度识图: {sub_title}")
        return str(sub_title)
    except requests.exceptions.RequestException as e:
        print(f"HTTP request error: {e}")
        return "百度识图未提取到有效信息"
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return "百度识图未提取到有效信息"
    except KeyError as e:
        print(f"Data extraction error: {e}")
        return "百度识图未提取到有效信息"
    except Exception as e:
        print(f"Error in extra_baidu_image_result: {e}")
        return "百度识图未提取到有效信息"