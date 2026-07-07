from crawl4ai import BrowserConfig, CrawlerRunConfig, DefaultMarkdownGenerator, PruningContentFilter, AsyncWebCrawler
import time
import asyncio
from typing import Optional, Tuple
# 1. 仅初始化一次的浏览器配置
GLOBAL_BROWSER_CONFIG = BrowserConfig(
    browser_type="chromium",
    headless=True,
    text_mode=True,
    verbose=True
)

# 2. 仅初始化一次的爬取配置（包含 markdown 生成器/内容过滤器）
GLOBAL_RUN_CONFIG = CrawlerRunConfig(
    markdown_generator=DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(
            threshold=0.7,
            threshold_type="fixed"
        ),
        options={"ignore_links": True, "ignore_images": True}
    ),
    verbose=True
)

global_crawler: Optional[AsyncWebCrawler] = None
# 单例对象（全程保留，仅停止/重启内部浏览器）
global_crawler: Optional[AsyncWebCrawler] = None
global_browser = None  # 手动持有浏览器实例

# ===================== 单例管理函数（核心修改） =====================
async def init_global_crawler():
    """初始化单例（首次创建/程序启动时调用）"""
    global global_crawler, global_browser
    if global_crawler is None:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        global_crawler = AsyncWebCrawler(config=GLOBAL_BROWSER_CONFIG)
    # 启动浏览器进程（无论是否首次，都确保启动）
    await global_crawler.start()
    # 保存浏览器实例（用于后续手动关闭）
    if hasattr(global_crawler, '_renderer') and hasattr(global_crawler._renderer, 'browser'):
        global_browser = global_crawler._renderer.browser
    print("$$$",global_crawler)

async def reinit_global_crawler():
    """复用单例，重新启动浏览器（循环后/暂停后调用）"""
    global global_crawler
    if global_crawler is None:
        await init_global_crawler()  # 未初始化则先初始化
    else:
        # 重新启动浏览器进程（无需重建Crawler对象）
        await global_crawler.start()
        global global_browser
        if hasattr(global_crawler, '_renderer') and hasattr(global_crawler._renderer, 'browser'):
            global_browser = global_crawler._renderer.browser

async def close_global_crawler():
    """临时关闭浏览器进程（保留单例，后续可重启）"""
    global global_browser
    # 1. 停止Crawler的浏览器实例
    if global_crawler is not None:
        await global_crawler.close()
    # 2. 强制关闭浏览器子进程
    if global_browser is not None:
        try:
            await global_browser.close()
        except Exception:
            pass
        global_browser = None
    # 3. 等待子进程退出（Windows必需）
    await asyncio.sleep(1.0)

async def destroy_global_crawler():
    """彻底销毁单例（程序最终退出时调用）"""
    global global_crawler, global_browser
    # 先临时关闭浏览器
    await close_global_crawler()
    # 彻底销毁单例
    global_crawler = None
    global_browser = None
    # 清理异步任务
    for task in asyncio.all_tasks():
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
#！！！！！ 确保在调用时已经初始化。批量调用时需要修正 ！！！！！        

