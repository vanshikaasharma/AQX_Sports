# builds player photo urls from basketball reference (run once)

import json
import os
import time

import requests
from bs4 import BeautifulSoup

PHOTO_PATH = "data/player_photos.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


# scrape player headshot urls from recent seasons
def build_photos(start=2018, end=2024):
    mapping = {}
    for year in range(start, end + 1):
        url = "https://www.basketball-reference.com/leagues/NBA_{}_per_game.html".format(year)
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.encoding = "utf-8"
        html = resp.text.replace("<!--", "").replace("-->", "")
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table", id="per_game_stats")
        if table is None:
            continue
        for row in table.find_all("tr"):
            link = row.find("a", href=True)
            if link is None or "/players/" not in link["href"]:
                continue
            name = link.get_text(strip=True)
            slug = link["href"].split("/")[-1].replace(".html", "")
            mapping[name] = (
                "https://www.basketball-reference.com/req/202106291/images/headshots/"
                + slug + ".jpg"
            )
        print("year", year, "total", len(mapping))
        time.sleep(3.5)
    return mapping


def main():
    os.makedirs("data", exist_ok=True)
    photos = build_photos()
    with open(PHOTO_PATH, "w", encoding="utf-8") as f:
        json.dump(photos, f)
    print("saved", len(photos), "photos to", PHOTO_PATH)


if __name__ == "__main__":
    main()
