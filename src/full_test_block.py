import asyncio
import os
import re
from playwright.async_api import async_playwright

urls = [
    "https://www.vttresearch.com/en/news-and-stories/customer-stories/forcit-making-civil-explosives-production-more-sustainable-and-cost-efficient",
    # Add 1–2 more URLs from your earlier logs for debugging
]

os.makedirs("raw_articles", exist_ok=True)


async def scrape_and_save_raw_html():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=200)
        context = await browser.new_context()
        page = await context.new_page()

        for url in urls:
            print(f"🔗 Visiting: {url}")
            try:
                await page.goto(url, timeout=20000)
                await page.wait_for_load_state("networkidle")
                await asyncio.sleep(3)

                html = await page.content()
                print(f"📄 Got {len(html)} characters from {url}")

                fname = re.sub(r"[^a-zA-Z0-9]", "_", url[-60:])[:50] + ".html"
                fpath = os.path.join("raw_articles", fname)
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(html)
                print(f"💾 Saved: {fpath}")

            except Exception as e:
                print(f"❌ Failed: {e}")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(scrape_and_save_raw_html())
