import time
from pathlib import Path

from playwright.sync_api import sync_playwright
from bedrock_agentcore.tools.browser_client import browser_session

REGION = "us-east-1"
URL = "https://www.udacity.com"

SCREENSHOT_DIR = Path("Screenshots")
SCREENSHOT_DIR.mkdir(exist_ok=True)


def main():
    print("Starting AgentCore Browser session...")

    with browser_session(REGION) as client:
        print("SUCCESS: Browser session started")
        print("SESSION ID:", client.session_id)

        ws_url, headers = client.generate_ws_headers()

        print("WebSocket URL generated:", bool(ws_url))
        print("Headers generated:", bool(headers))

        with sync_playwright() as p:
            print("Connecting Playwright to AgentCore browser...")

            browser = p.chromium.connect_over_cdp(
                ws_url,
                headers=headers
            )

            print("SUCCESS: Playwright connected!")

            # IMPORTANT:
            # AWS recommends using the existing/default context.
            context = browser.contexts[0]

            if context.pages:
                page = context.pages[0]
            else:
                page = context.new_page()

            print("Opening:", URL)

            page.goto(URL, wait_until="domcontentloaded", timeout=120000)

            print("PAGE LOADED")

            title = page.title()
            print("TITLE:", title)

            print("URL:", page.url)

            # Grab visible text
            text = page.locator("body").inner_text(timeout=30000)

            print("\nVISIBLE TEXT SAMPLE:")
            print(text[:2000])

            # Screenshot 1
            screenshot1 = SCREENSHOT_DIR / "udacity_agentcore.png"
            page.screenshot(
                path=str(screenshot1),
                full_page=True
            )

            print("\nSCREENSHOT SAVED:")
            print(screenshot1.resolve())

            # Take a second screenshot after a short wait
            time.sleep(3)

            screenshot2 = SCREENSHOT_DIR / "udacity_agentcore_2.png"
            page.screenshot(
                path=str(screenshot2),
                full_page=True
            )

            print("SECOND SCREENSHOT SAVED:")
            print(screenshot2.resolve())

            print("\nKEEPING SESSION ALIVE FOR 60 SECONDS...")
            print("You can inspect the browser session in AWS during this time.")

            time.sleep(60)
            time.sleep(3)

            screenshot2 = SCREENSHOT_DIR / "udacity_agentcore_2.png"

            page.screenshot(
                path=str(screenshot2),
                full_page=True
            )

            print("SECOND SCREENSHOT SAVED:")
            print(screenshot2.resolve())

            browser.close()

        print("Browser session finished.")


if __name__ == "__main__":
    main()