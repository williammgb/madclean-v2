"""The Profile view in a real browser: every column as a card, two per row, with findings.

`./run ui` runs this against the app it started, passing the app's address:

    python tests/ui/profile_grid.py http://localhost:3000/

It uploads the hospital benchmark table, opens Profile, and checks that there is one card per
column, two side by side at 1440 px, that ZipCode lists odd values such as typed-in "x" digits,
and that "With findings" hides the clean columns. A screenshot goes to
.work/screens/profile-grid.png. Any error in the page's console fails it.
"""

import sys
from pathlib import Path

import pandas as pd
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[2]
SCREENS = ROOT / ".work" / "screens"
DATASET = ROOT / "data" / "benchmark_datasets" / "hospital_dirty.csv"
READ_CARDS = """() => [...document.querySelectorAll('.prof-card')].map((card) => ({
  name: card.querySelector('.prof-head h2').textContent.trim(),
  top: Math.round(card.getBoundingClientRect().top),
  left: Math.round(card.getBoundingClientRect().left),
  finds: [...card.querySelectorAll('.find-head h3')].map((h) => h.textContent.trim()),
  odd: [...card.querySelectorAll('.find .val b')].map((b) => b.textContent.trim()),
}))"""


def main(url: str) -> int:
    SCREENS.mkdir(parents=True, exist_ok=True)
    columns = list(pd.read_csv(DATASET, nrows=1).columns)
    errors: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge")
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url)

        page.locator("#dataset_upload input[type=file]").set_input_files(str(DATASET))
        expect(page.get_by_role("button", name="Run cleaning")).to_be_enabled(timeout=120_000)
        page.locator(".nav-item", has_text="Profile").click()
        expect(page.locator(".prof-card")).to_have_count(len(columns), timeout=60_000)

        cards = page.evaluate(READ_CARDS)
        assert [c["name"] for c in cards] == columns, [c["name"] for c in cards]
        assert cards[0]["top"] == cards[1]["top"] and cards[0]["left"] < cards[1]["left"], cards[:2]
        assert cards[2]["top"] > cards[0]["top"], "the third card is not on a second row"
        print(f"ui: {len(cards)} cards, two per row")

        zipcode = next(c for c in cards if c["name"] == "ZipCode")
        assert any(f.startswith("Odd values") for f in zipcode["finds"]), zipcode
        assert any("x" in v for v in zipcode["odd"]), zipcode
        print(f"ui: ZipCode findings {zipcode['finds']}, odd values {zipcode['odd'][:4]}")
        page.screenshot(path=str(SCREENS / "profile-grid.png"))

        with_finds = [c["name"] for c in cards if c["finds"]]
        assert 0 < len(with_finds) < len(cards), with_finds
        page.get_by_role("button", name="With findings").click()
        expect(page.locator(".prof-card")).to_have_count(len(with_finds))
        page.get_by_role("button", name="All columns").click()
        expect(page.locator(".prof-card")).to_have_count(len(columns))
        print(f"ui: 'With findings' shows {len(with_finds)} of {len(cards)} cards")
        browser.close()
    if errors:
        print("ui: console errors:\n" + "\n".join(errors))
        return 1
    print("ui: profile grid checks passed, no console errors")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:3000/"))
