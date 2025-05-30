import asyncio
from playwright.async_api import async_playwright
from urllib.parse import urljoin, urlparse
import re

BASE_URL = "https://www.vttresearch.com/en/news-and-stories/customer-stories?page={}"
ALLOWED_DOMAIN = urlparse(BASE_URL).netloc
KEYWORDS = ["vtt", "collaboration", "innovation", "research", "project"]
FILENAME = "vtt_playwright_d1.txt"

visited = set()


def url_allowed(url):
    parsed = urlparse(url)
    return parsed.netloc == ALLOWED_DOMAIN and url not in visited


async def scrape_page(page, url):
    print(f"🔗 Visiting: {url}")
    try:
        await page.goto(url, timeout=15000)
        await page.wait_for_load_state("networkidle")
        text = await page.text_content("body")
        return text
    except Exception as e:
        print(f"⚠️ Error loading {url}: {e}")
        return None


async def extract_story_links(page, base_url):
    cards = await page.query_selector_all("div.node__title")
    links = []
    for card in cards:
        anchor = await card.query_selector("a[href]")
        if anchor:
            href = await anchor.get_attribute("href")
            if href:
                abs_url = urljoin(base_url, href)
                if url_allowed(abs_url):
                    links.append(abs_url)
    return list(set(links))  # Deduplicate


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        page_num = 1
        while True:
            index_url = BASE_URL.format(page_num)
            print(f"\n📄 Checking index page {index_url}")
            await page.goto(index_url)
            await page.wait_for_load_state("networkidle")

            links = await extract_story_links(page, index_url)
            if not links:
                print(f"🚫 No more links found on page {page_num}. Stopping.")
                break

            for link in links:
                if link in visited:
                    continue
                visited.add(link)
                content = await scrape_page(page, link)
                if not content:
                    continue

                lower_text = content.lower()
                if any(kw in lower_text for kw in KEYWORDS):
                    with open(FILENAME, "a", encoding="utf-8") as f:
                        f.write(f"\n\n=== {link} ===\n{content.strip()}\n")

            page_num += 1

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
