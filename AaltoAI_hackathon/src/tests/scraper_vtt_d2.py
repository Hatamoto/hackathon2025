import os
import re
import asyncio
from playwright.async_api import async_playwright
from urllib.parse import urljoin

BASE_PAGE_URL = "https://www.vttresearch.com/en/news-and-stories/customer-stories?page={}"
BASE_DOMAIN = "https://www.vttresearch.com"
KEYWORDS = ["vtt", "collaboration", "innovation", "research", "project"]

visited = set()


async def extract_story_links(page, base_url):
    try:
        await page.goto(base_url, timeout=20000)
        print(f"🧪 Loaded page: {base_url}")

        # Scroll to trigger lazy loading
        for _ in range(5):
            await page.mouse.wheel(0, 1000)
            await asyncio.sleep(1)

        html = await page.content()
        with open(f"debug_page_{base_url.split('=')[-1]}.html", "w", encoding="utf-8") as f:
            f.write(html)
        print(f"📄 Saved HTML snapshot for {base_url}")

        # Look for the divs with article titles
        title_divs = await page.query_selector_all("div.node__title")
        print(f"🔍 Found {len(title_divs)} title divs")

        links = []
        for div in title_divs:
            anchor = await div.query_selector("a[href]")
            if anchor:
                href = await anchor.get_attribute("href")
                if href:
                    full_url = urljoin(BASE_DOMAIN, href)
                    if full_url not in visited:
                        print(f"✅ Found story link: {full_url}")
                        links.append(full_url)

        return links

    except Exception as e:
        print(f"⚠️ Failed to load or timeout: {base_url} — {e}")
        return []


# Ensure output folder exists
os.makedirs("raw_articles", exist_ok=True)


async def scrape_story_text(page, url):
    print(f"🔗 Visiting: {url}")
    try:
        await page.goto(url, timeout=20000)
        await page.wait_for_load_state("networkidle")
        await asyncio.sleep(3)

        html = await page.content()
        if not html or len(html.strip()) < 50:
            print(f"⚠️ Very short or empty HTML for: {url}")

        # Safe filename based on URL hash
        filename = re.sub(r"[^a-zA-Z0-9]", "_", url[-60:])[:50] + ".html"
        path = os.path.join("raw_articles", filename)

        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"💾 Saved HTML to: {path}")
        return html

    except Exception as e:
        print(f"❌ Failed to scrape {url}: {e}")
        return None


async def main():
    async with async_playwright() as p:
        # Visual mode helps here
        browser = await p.chromium.launch(headless=False, slow_mo=250)
        context = await browser.new_context()
        page = await context.new_page()

        page_num = 1
        while True:
            current_url = BASE_PAGE_URL.format(page_num)
            print(f"\n📄 Crawling index page {current_url}")
            links = await extract_story_links(page, current_url)

            if not links:
                print("🚫 No more stories found, stopping.")
                break

            for link in links:
                if link in visited:
                    continue
                visited.add(link)
                content = await scrape_story_text(page, link)

                if not content:
                    print(f"⚠️ No content from {link}")
                    continue

                print(f"📄 Scraped {len(content)} characters from {link}")

                # Always save for now
                with open("vtt_stories_scraped.txt", "a", encoding="utf-8") as f:
                    f.write(f"\n\n=== {link} ===\n{content.strip()}\n")

                # Optional: log if keywords matched
                if any(k in content.lower() for k in KEYWORDS):
                    print("🔑 Matched keywords.")
                else:
                    print("🕵️ No keyword match.")

            page_num += 1

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
