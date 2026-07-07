from PicImageSearch import BaiDu, Network
from PicImageSearch.model import BaiDuResponse
from bs4 import BeautifulSoup
import requests
import json
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, LLMConfig, DefaultMarkdownGenerator, BrowserConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter
from crawl4ai.content_filter_strategy import LLMContentFilter, PruningContentFilter, RelevantContentFilter
from pydantic import BaseModel
from typing import Optional


class CrawlRequest(BaseModel):
    url: Optional[str] = None
    prefix: Optional[str] = None
    if_llm: Optional[bool] = None
    base_url: Optional[str] = True
    api_key: Optional[str] = True
    provider: Optional[str] = True
    model_id: Optional[str] = True


MAX_URLS = 1
BLACKLIST_DOMAINS = []


def extra_baidu_image_result(url):
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
            return

        # 结束：从起始位置后找最近的 "];", 因为数组一定以 "];"" 结束
        end_idx = script_content.find("];", start_idx)
        if end_idx == -1:
            print("❌ 未找到 window.cardData 的结束标记（];）")
            # 调试：打印找到的 script 内容片段（帮助确认格式）
            print(f"\n调试：找到的 script 内容片段（start_idx附近）：")
            print(script_content[max(0, start_idx - 100): start_idx + 200] + "...")
            return

        # 提取完整 JSON 字符串（包含 []）
        card_data_str = script_content[start_idx + len(start_marker) - 1: end_idx + 1]

        # 5. 解析 JSON 并提取目标值
        card_data = json.loads(card_data_str)
        sub_title = card_data[0]["tplData"]["pano"]["list"][0]["subTitle"]
        print(f"✅ 成功提取 subTitle：{sub_title}")
        return str(sub_title)
    except:
        return ""


def get_web_urls(resp: BaiDuResponse, top_k: int = MAX_URLS):
    urls = []
    baidu_result = extra_baidu_image_result(resp.url)
    for item in resp.raw:
        if hasattr(item, "url") and item.url:
            if any(domain in item.url for domain in BLACKLIST_DOMAINS):
                continue
            urls.append(item.url)
        if len(urls) >= top_k:
            break
    return urls, baidu_result


async def demo_async(file, top_k=MAX_URLS):
    async with Network() as client:
        baidu = BaiDu(client=client)
        resp = await baidu.search(file=file)
        print("resp_url:", resp.url)
        print("resp_raw", resp.raw)
        print(len(resp.raw))
        for a in resp.raw:
            if a.url:
                print("resp_raw_url", a.url)
        return get_web_urls(resp, top_k)


async def crawl_site(request: CrawlRequest, baidu_res: str, query: str):
    """深度解析 website"""
    site, prefix, if_llm, model_id, provider, base_url, api_token = (
        request.url,
        request.prefix,
        request.if_llm,
        request.model_id,
        request.provider,
        request.base_url,
        request.api_key
    )
    content_filter: RelevantContentFilter
    if if_llm:
        content_filter = LLMContentFilter(
            llm_config=LLMConfig(provider=f"{provider}/{model_id}", api_token=api_token, base_url=base_url),
            instruction=f"""
                # 重点在于提取与以下问题相关的核心内容。
                问题：
                - {query}
                参考：
                - 根据外部数据源查询结果，用户提供的图片中可能是{baidu_res}
                排除：
                - 导航元素
                - 侧边栏
                - 页脚内容
                要求：
                - 只提取与用户问题高度相关的有效参考信息，与用户问题无关的内容不需要提取出来
                将输出格式化为清晰的Markdown格式，包含恰当的换行符和标题
                """,
            chunk_token_threshold=4096,
            verbose=True
        )
    else:
        content_filter = PruningContentFilter(
            threshold=0.9,
            threshold_type="fixed"
        )
    md_generator = DefaultMarkdownGenerator(
        content_filter=content_filter,
        options={"ignore_links": True, "ignore_images": True}
    )

    browser_conf = BrowserConfig(
        browser_type="chromium",
        headless=True,
        text_mode=True
    )

    prefix_filter = URLPatternFilter(
        patterns=[f"{prefix}*"]
    )
    filter_chain = FilterChain([prefix_filter])

    # 深度爬取策略
    bfsstrategy = BFSDeepCrawlStrategy(
        max_depth=0,  # 爬取深度
        include_external=False,  # 限制同域
        max_pages=200,  # 爬取的最大页面数
        filter_chain=filter_chain
    )

    # 爬取配置
    config = CrawlerRunConfig(
        deep_crawl_strategy=bfsstrategy,
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=True,
        stream=True,
        markdown_generator=md_generator,
    )

    async with AsyncWebCrawler(config=browser_conf) as crawler:
        try:
            async for result in await crawler.arun(site, config=config):
                try:
                    print(f"URL: {result.url}")

                    # 检查result.markdown是否存在且不为None
                    if result.markdown is None:
                        print(f"URL {result.url} 的markdown内容为空，跳过处理")
                        continue
                    print(f"提取前的内容:{result.markdown}")
                    print("==================================")

                    # 检查fit_markdown是否存在且不为空
                    if not hasattr(result.markdown, 'fit_markdown') or result.markdown.fit_markdown is None:
                        print(f"URL {result.url} 的fit_markdown内容为空，跳过处理")
                        continue

                    # 检查内容是否为空字符串
                    if not result.markdown.fit_markdown.strip():
                        print(f"URL {result.url} 的内容为空，跳过处理")
                        continue

                    print(f"成功解析：{result.url}")
                    print(f"有效摘要信息：{result.markdown.fit_markdown}")

                except Exception as e:
                    error_msg = f"处理URL {result.url} 时发生错误: {str(e)}"
                    print(error_msg)
                    print("跳过此URL，继续处理下一个...")
                    continue

        except Exception as e:
            error_msg = f"爬虫运行时发生错误: {str(e)}"
            print(error_msg)
            print("爬虫任务中断，但程序继续运行...")
        print("网址解析任务处理完成")


async def crawl_all_site(urls: list, baidu_res: str, request: CrawlRequest, query: str):
    tasks = []
    for url in urls:
        param = CrawlRequest(
            url=url,  # 为每个任务传递不同的URL
            prefix=request.prefix,
            if_llm=request.if_llm,
            model_id=request.model_id,
            provider=request.provider,
            base_url=request.base_url,
            api_key=request.api_key
        )
        tasks.append(crawl_site(param, baidu_res, query))
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    image_path = "../test/character.jpg"
    urls, baidu_res = asyncio.run(demo_async(image_path, top_k=MAX_URLS))
    print(f"百度搜索结果：{baidu_res}\n百度搜索url：{urls}")

    query = "图片中的人物是谁？饰演的什么角色？"
    param = CrawlRequest(
        prefix="https://",
        if_llm=False,
        base_url=os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1'),
        api_key=os.getenv('DEEPSEEK_API_KEY', ''),
        provider="deepseek",
        model_id="deepseek-chat"
    )

    asyncio.run(crawl_all_site(urls, baidu_res, param, query))

