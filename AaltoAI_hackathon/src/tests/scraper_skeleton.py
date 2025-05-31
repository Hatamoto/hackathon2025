import asyncio
from playwright.async_api import async_playwright
from urllib.parse import urljoin, urlparse
import re

BASE_URL = "https://www.vttresearch.com/en/news-and-stories/customer-stories?page=1"
ALLOWED_DOMAIN = urlparse(BASE_URL).netloc
KEYWORDS = ["vtt", "collaboration", "innovation", "research", "project"]
MAX_DEPTH = 2

visited = set()
queue = [(BASE_URL, 0)]


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


async def extract_links(page, base_url):
    anchors = await page.query_selector_all("a[href]")
    links = []
    for a in anchors:
        href = await a.get_attribute("href")
        if href:
            abs_url = urljoin(base_url, href)
            if url_allowed(abs_url):
                links.append(abs_url)
    return links


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        while queue:
            current_url, depth = queue.pop(0)
            if depth > MAX_DEPTH:
                continue

            visited.add(current_url)
            content = await scrape_page(page, current_url)
            if not content:
                continue

            lower_text = content.lower()
            if any(kw in lower_text for kw in KEYWORDS):
                with open("vtt_playwright_scraped.txt", "a", encoding="utf-8") as f:
                    f.write(f"\n\n=== {current_url} ===\n{content.strip()}\n")

            links = await extract_links(page, current_url)
            for link in links:
                if link not in visited:
                    queue.append((link, depth + 1))

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
