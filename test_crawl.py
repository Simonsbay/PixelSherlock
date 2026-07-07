import asyncio
from crawl4ai import *
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://www.baidu.com",
        )
        print(result.markdown)

if __name__ == "__main__":
    asyncio.run(main())