"""Ticket 15 in a real browser: the top bar and the Pipeline show the model calls that are open.

`./run ui` runs this against the app it started with the offline demo model
(MADCLEAN_DEMO_LLM_SECONDS=5), passing the app's address:

    python tests/ui/ticket_15.py http://localhost:3000/

It uploads a small dataset, starts a run, and checks during the run that the top bar shows
`<model> · N calls · m:ss` counting up with a pulsing dot and that the Pipeline shows a pulsing
`Recommender m:ss` chip on a row whose result reads Running; after the run, that the line and
the chips are gone. Screenshots go to .work/screens/ticket-15-*.png. Any error in the page's
console fails it.
"""

import re
import sys
import time
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[2]
SCREENS = ROOT / ".work" / "screens"
DATASET = ROOT / ".work" / "tmp" / "ticket-15-small.csv"
LINE = re.compile(r"^demo · (\d+) calls? · (\d+):(\d\d)$")
HOVER = re.compile(r"^(Recommender|Coder|Validator)( \d+)? · .+ · \d+:\d\d$")
CHIP = re.compile(r"^Recommender\s*\d+:\d\d$")
TOPBAR_CLIP = {"x": 0, "y": 0, "width": 1440, "height": 60}
# Everything the checks look at, read in one go: the top bar's API line and every chip with a clock.
READ_PAGE = """() => {
  const squash = (el) => el ? el.textContent.replace(/\\s+/g, ' ').trim() : '';
  const line = document.querySelector('.topbar .status .api');
  const dot = line && line.querySelector('.dot.pulse');
  const chips = [...document.querySelectorAll('.task .step')].filter((s) => s.querySelector('.secs')).map((s) => {
    const row = s.closest('.task');
    const pulse = s.querySelector('.pulse');
    return {
      text: squash(s), cls: s.className, pulse: pulse ? getComputedStyle(pulse).animationName : '',
      result: squash(row.querySelector('.result')), steps: row.querySelectorAll('.step').length,
    };
  });
  return {
    line: line ? squash(line) : null, title: line ? line.getAttribute('title') : null,
    dot: dot ? getComputedStyle(dot).animationName : '', ring: !!(line && line.querySelector('.ring')),
    chips, status: squash(document.querySelector('.topbar .status .chip')),
  };
}"""


def text_of(locator) -> str:
    """The element's text with its whitespace collapsed, as a person reads it."""
    return " ".join((locator.text_content() or "").split())


def seconds_of(text: str) -> int | None:
    match = LINE.match(text.strip())
    return None if match is None else int(match.group(2)) * 60 + int(match.group(3))


def main(url: str) -> int:
    SCREENS.mkdir(parents=True, exist_ok=True)
    DATASET.parent.mkdir(parents=True, exist_ok=True)
    rows = "".join(f"item {i},{['Amsterdam', 'Utrecht', 'Delft'][i % 3]},{i}.50\n" for i in range(24))
    DATASET.write_text("name,city,price\n" + rows, encoding="utf-8")
    errors: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge")
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url)

        page.locator("#dataset_upload input[type=file]").set_input_files(str(DATASET))
        run_button = page.get_by_role("button", name="Run cleaning")
        expect(run_button).to_be_enabled(timeout=90_000)
        run_button.click()
        print("ui: run started")

        # The Pipeline is open for the whole run; the top bar shows on every view. A demo call is
        # open for 5 s, so the page is read in one go each time rather than element by element.
        page.locator(".nav-item", has_text="Pipeline").click()
        recommender_chip = first_call_row = counted = local_seen = None
        previous = None
        deadline = time.monotonic() + 400
        while time.monotonic() < deadline:
            seen = page.evaluate(READ_PAGE)
            if seen["status"].startswith("Finished in"):
                break
            for chip in seen["chips"]:
                if (
                    recommender_chip is None
                    and CHIP.match(chip["text"])
                    and "running" in chip["cls"].split()
                    and chip["pulse"] == "pulse"
                ):
                    recommender_chip = chip
                    page.screenshot(path=str(SCREENS / "ticket-15-pipeline.png"))
                    print(f"ui: pipeline chip {chip['text']!r} ({chip['cls']}), row result {chip['result']!r}")
                if first_call_row is None and chip["steps"] == 1 and chip["result"] == "Running":
                    first_call_row = chip
                    print(f"ui: a row with only its open call: {chip['text']!r}, result {chip['result']!r}")
            line = seen["line"]
            if line and line.startswith("Working locally") and seen["ring"] and local_seen is None:
                local_seen = line
                print(f"ui: between calls the top bar reads {line!r} with a ring")
            now = seconds_of(line) if line else None
            if now is not None:
                assert seen["dot"] == "pulse", f"the dot does not pulse: {seen['dot']!r}"
                hover = seen["title"] or ""
                assert hover and all(HOVER.match(e) for e in hover.splitlines()), f"hover {hover!r}"
                if counted is None and previous is not None and now > previous[1] and line != previous[0]:
                    counted = (previous[0], line)
                    page.screenshot(path=str(SCREENS / "ticket-15-topbar.png"), clip=TOPBAR_CLIP)
                    print(f"ui: top bar counted up from {previous[0]!r} to {line!r}, hover {hover!r}")
                previous = (line, now)
            else:
                previous = None
            time.sleep(0.25)
        assert recommender_chip is not None, "the Pipeline never showed a pulsing Recommender m:ss chip"
        assert first_call_row is not None, "no row for a column whose first call was open"
        assert counted is not None, "the top bar never showed <model> · N calls · m:ss counting up"

        finished = page.locator(".topbar .status .chip", has_text="Finished in")
        expect(finished).to_be_visible(timeout=60_000)
        expect(page.locator(".topbar .api")).to_have_count(0)
        expect(page.locator(".task .step .secs")).to_have_count(0)
        print(f"ui: after the run the top bar reads {text_of(finished)!r} and the API line is gone")
        page.screenshot(path=str(SCREENS / "ticket-15-after.png"))
        browser.close()
    if errors:
        print("ui: console errors:\n" + "\n".join(errors))
        return 1
    print("ui: ticket 15 checks passed, no console errors")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:3000/"))
