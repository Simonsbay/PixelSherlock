from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import CrawlerRunConfig
import asyncio
from crawl4ai import BrowserConfig, CrawlerRunConfig, DefaultMarkdownGenerator, PruningContentFilter, AsyncWebCrawler
import time
import asyncio
from typing import Optional, Tuple
import time
URLS = [
    "http://news.sohu.com/a/611719908_638188",
    "http://baijiahao.baidu.com/s?id=1812075862080112623&wfr=spider&for=pc",
    "http://www.163.com/v/video/VN22RMTN5.html",
    "http://dy.163.com/article/CTJEIGPQ0524B9KK.html",
    "http://www.bilibili.com/video/BV1eY4y1n7tH",
    "http://www.meipian.cn/57xsi3wh",
    "http://www.jnzx.gov.cn/sl/hssl/2020-07-13/108271.html",
    "http://huaban.com/pins/2728814023/",
    "http://www.zhihu.com/zvideo/1664388852621107200",
    "http://stock.tuchong.com/video/detail?videoId=1832628849093903912",
    "http://mbd.baidu.com/newspage/data/dtlandingsuper?nid=dt_5038560911879848951",
    "http://www.bilibili.com/video/BV1KK4y1g7EF",
    "http://baijiahao.baidu.com/s?id=1848241360372006112&wfr=spider&for=pc",
    "http://www.bilibili.com/video/av61270426",
    "http://xsj.699pic.com/tupian/1nwq8w.html",
    "http://m.bilibili.com/video/av205855954",
    "http://m.v.qq.com/page/c/0/5/c0783caair5.html",
    "http://baijiahao.baidu.com/s?id=1832095037422944013&wfr=spider&for=pc",
    "http://www.meipian.cn/2xqy02nr",
    "http://club.autohome.com.cn/bbs/threadowner/cbaf87c122b28bc6/90751728-1.html"
]


async def main():
    browser_conf = BrowserConfig(
        browser_type="chromium",
        headless=True,
        text_mode=True,
        verbose=True
    )
    run_cfg = CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(
                threshold=0.7,
                threshold_type="fixed"
            ),
            options={"ignore_links": True, "ignore_images": True}
        ),
        verbose=True
    )
    s = time.time()
    async with AsyncWebCrawler(config=browser_conf) as crawler:
        results = await crawler.arun_many(URLS, config=run_cfg, max_concurrency=5)
        
        for url, res in zip(URLS, results):
            print(url, res.success, len(res.markdown))
            print(res)
    e = time.time()
    print(e-s)

asyncio.run(main())
