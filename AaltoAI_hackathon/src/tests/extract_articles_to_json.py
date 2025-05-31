import os
import json
import hashlib
from bs4 import BeautifulSoup
from bs4.element import Tag
from pathlib import Path
from typing import Optional, Dict, Any

SOURCE_DIR = "raw_articles"
OUTPUT_FILE = "vtt_articles.jsonl"


def extract_text_from_html(filepath: Path) -> Optional[Dict[str, Any]]:
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return None

    with filepath.open("r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    # --- Extract title ---
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if isinstance(
        title_tag, Tag) else "Untitled"

    # --- Extract body content using partial match on class ---
    body_div = soup.find("div", class_="field--name-body")
    if not isinstance(body_div, Tag):
        print(f"⚠️ No body content found in: {filepath.name}")
        return None

    paragraphs = [p.get_text(strip=True)
                  for p in body_div.find_all("p") if p.get_text(strip=True)]
    full_text = "\n\n".join(paragraphs)

    if not full_text:
        print(f"⚠️ No valid text in: {filepath.name}")
        return None

    # --- Extract date ---
    date = None
    meta_tag = soup.find("meta", {"property": "article:published_time"})
    if isinstance(meta_tag, Tag):
        content_attr = meta_tag.get("content")
        if isinstance(content_attr, str):
            date = content_attr[:10]

    # --- Build record ---
    text_hash = hashlib.sha1(full_text.encode()).hexdigest()[:8]
    innovation_id = f"vtt_{text_hash}"
    summary = "\n".join(paragraphs[:2])

    return {
        "innovation_id": innovation_id,
        "core_summary": summary,
        "participants": ["VTT"],
        "descriptions": [{
            "source": str(filepath),
            "text": full_text,
            "date": date
        }]
    }


def main() -> None:
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with output_path.open("w", encoding="utf-8") as out_file:
        for file in Path(SOURCE_DIR).glob("*.html"):
            print(f"📄 Parsing: {file.name}")
            data = extract_text_from_html(file)
            if data:
                json.dump(data, out_file, ensure_ascii=False)
                out_file.write("\n")
                count += 1
            else:
                print(f"⚠️ Skipped: {file.name}")

    print(f"\n✅ Extracted {count} articles to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
