"""ColonAI — verify the LIVE Space is producing real predictions.

Uses Playwright (headless Chromium) to actually render the page,
click into "AI Analysis" so the model load is triggered, then read
the "ℹ︎ Why?" expander in the sidebar to get the EXACT reason if
the model failed to load.

Saves a full-page screenshot AND a separate sidebar crop.
Writes a structured result to outputs/live_status.json.
"""
from __future__ import annotations
import sys, time, json, re
from datetime import datetime, timezone
from pathlib import Path

URL = "https://yuvraj2319-colonai.hf.space/"
OUT_DIR = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)


def check() -> dict:
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        return {"status": "UNKNOWN", "reason": "playwright not installed"}

    result = {"url": URL, "checked_at": datetime.now(timezone.utc).isoformat(),
              "status": "UNKNOWN", "reason": "", "diagnostic": {}}

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1400, "height": 1000})
        page = ctx.new_page()

        # 1. Navigate
        try:
            page.goto(URL, wait_until="domcontentloaded", timeout=45_000)
        except Exception as e:
            result.update(status="DOWN", reason=f"{type(e).__name__}: {e}")
            browser.close(); return result

        # 2. Wait for the sidebar to render
        try:
            page.wait_for_selector('[data-testid="stSidebar"]', timeout=60_000)
        except PWTimeout:
            result.update(status="UNKNOWN", reason="sidebar never appeared")
            shot = OUT_DIR / f"live_status_{int(time.time())}.png"
            page.screenshot(path=str(shot), full_page=True)
            result["screenshot"] = str(shot)
            browser.close(); return result

        # 3. Click "AI Analysis" twice with backoff — Streamlit nav can race
        # with the WebSocket warm-up on the first click.
        for attempt in range(3):
            try:
                btn = page.locator('button:has-text("AI Analysis")').first
                if btn.count() > 0:
                    btn.click(timeout=10_000)
                    time.sleep(3)   # let Streamlit rerun
            except Exception as e:
                result[f"click_attempt_{attempt}_error"] = str(e)
            # If patient info is required first, this won't change the page —
            # in that case fall through to a demo-case button click below.
            time.sleep(1)
            html_check = page.content().lower()
            if "carefully analysing your case" in html_check or \
               "ai pipeline" in html_check or "loading ai pipeline" in html_check:
                break

        # If we didn't reach the analysis page, try a demo-case button (Patient Info
        # has "Load Case A · Sigmoid Polyp" which auto-fills + advances)
        if "carefully analysing" not in page.content().lower():
            for label in ["Load Case A", "Use a demo case", "Try a demo case"]:
                try:
                    case_btn = page.locator(f'button:has-text("{label}")').first
                    if case_btn.count() > 0:
                        case_btn.click(timeout=5_000)
                        time.sleep(3)
                        # Now try AI Analysis again
                        btn = page.locator('button:has-text("AI Analysis")').first
                        if btn.count() > 0:
                            btn.click(timeout=8_000)
                            time.sleep(3)
                        break
                except Exception:
                    pass

        # Wait for the load_ai_system call to settle — up to 90s total
        for _ in range(45):
            time.sleep(2)
            html = page.content().lower()
            if "ai pipeline ready" in html and "checkpoint loaded" in html:
                result["status"] = "READY"; break
            if "model load failed" in html or "demo mode" in html or \
               "ai system not available" in html:
                result["status"] = "DEMO_MODE"; break
            # If the spinner is still up, keep waiting
            if "loading ai pipeline" in html: continue

        # 4. If demo mode, try to find the "ℹ︎ Why?" expander and click it
        if result["status"] == "DEMO_MODE":
            try:
                exp = page.locator('text=ℹ︎ Why?').first
                if exp.count() > 0:
                    exp.click(timeout=5_000)
                    time.sleep(2)
                # Capture sidebar text
                sidebar = page.locator('[data-testid="stSidebar"]')
                sb_text = sidebar.inner_text(timeout=5_000)
                result["sidebar_text"] = sb_text[:3000]
            except Exception as e:
                result["sidebar_error"] = str(e)

            # Also try to grab the main-area error banner
            try:
                error_box = page.locator('[data-baseweb="notification"]').all_inner_texts()
                if error_box:
                    result["error_banners"] = error_box[:5]
            except Exception:
                pass

            # Capture every visible st.error message
            try:
                err_text = page.locator('.stAlert').all_inner_texts()
                if err_text:
                    result["alerts"] = err_text[:5]
            except Exception:
                pass

            # Look for our [CHECKPOINT] / [STARTUP] diagnostic strings
            html = page.content()
            for needle in ["CHECKPOINT", "STARTUP", "checkpoint", "preexisting",
                           "no_env_var", "downloaded", "HF repo:", "stage:"]:
                idx = html.find(needle)
                if idx > 0:
                    snippet = re.sub(r"<[^>]+>", " ",
                                     html[max(0, idx-100):idx+300])
                    result["diagnostic"][needle] = " ".join(snippet.split())[:300]

        # 5. Always screenshot
        shot = OUT_DIR / f"live_status_{int(time.time())}.png"
        try:
            page.screenshot(path=str(shot), full_page=True)
            result["screenshot"] = str(shot)
        except Exception:
            pass

        # 6. Sidebar-only screenshot (easier to read at scale)
        try:
            sb_shot = OUT_DIR / f"live_status_sidebar_{int(time.time())}.png"
            page.locator('[data-testid="stSidebar"]').screenshot(path=str(sb_shot))
            result["sidebar_screenshot"] = str(sb_shot)
        except Exception:
            pass

        browser.close()
    return result


def main():
    print(f"Checking live URL: {URL}")
    r = check()
    print(json.dumps({k: v for k, v in r.items() if k != "diagnostic"}, indent=2))
    if r.get("diagnostic"):
        print("\n[diagnostic snippets found in page HTML]")
        for k, v in r["diagnostic"].items():
            print(f"  {k}: {v[:200]}")
    out = OUT_DIR / "live_status.json"
    out.write_text(json.dumps(r, indent=2))
    print(f"\n  → wrote {out}")

    code = {"READY": 0, "DEMO_MODE": 1, "UNKNOWN": 2, "DOWN": 3}.get(r["status"], 2)
    sys.exit(code)


if __name__ == "__main__":
    main()
