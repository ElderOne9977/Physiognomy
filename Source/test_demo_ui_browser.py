"""Browser checks against an already-running mock Gradio preview (no real faces/models)."""
import json
import os
import tempfile
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright
import sys
sys.path.insert(0, str(Path(__file__).parent))
from demo_gradio import GENERATOR_LABELS

OUT = Path(__file__).parent / "experiments" / "ui_reading_20261014"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checks, errors, console_errors = [], [], []
    def check(name, condition):
        checks.append({"name": name, "pass": bool(condition)})
        print(("PASS " if condition else "FAIL ") + name, flush=True)
    with tempfile.TemporaryDirectory(prefix="physio_ui_browser_") as td, sync_playwright() as p:
        image = Path(td) / "synthetic.png"
        Image.new("RGB", (320, 320), (223, 234, 227)).save(image)
        browser = p.chromium.launch(executable_path=os.environ.get("PHYSIO_UI_BROWSER_EXECUTABLE"))
        context = browser.new_context(viewport={"width": 1440, "height": 1000}, permissions=["clipboard-read", "clipboard-write"])
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
        page.goto("http://127.0.0.1:7862", wait_until="networkidle")
        page.locator("#physio-app").wait_for()
        switch = page.get_by_role('switch', name='Chế độ tối')
        switch.click()
        page.wait_for_function("document.querySelector('[role=switch]').getAttribute('aria-checked') === 'true'")
        check('theme switch enables dark mode', page.locator('body').evaluate("e => e.classList.contains('dark')"))
        page.reload(wait_until='networkidle')
        check('theme persists on reload', switch.get_attribute('aria-checked') == 'true')
        switch.focus()
        page.keyboard.press('Space')
        check('keyboard switches to light', switch.get_attribute('aria-checked') == 'false')
        check('geometry illustration accessible', page.get_by_role('img', name='Minh họa các điểm mốc hình học khuôn mặt').is_visible())
        check('illustration clearly labeled', page.get_by_text('Sơ đồ minh họa, không phải kết quả đo từ ảnh của bạn.', exact=True).is_visible())
        check('header copy removed', page.locator('.product-label, .theme-value, .feature-guide .eyebrow').count() == 0)
        check('theme icons switch with state', switch.locator('.theme-sun').is_visible() and not switch.locator('.theme-moon').is_visible())
        page.set_viewport_size({'width':375,'height':1000})
        check('theme switch inside header on mobile', switch.evaluate("e => {const b=e.getBoundingClientRect(), h=e.closest('header').getBoundingClientRect(); return b.left >= h.left && b.right <= h.right && b.top >= h.top && b.bottom <= h.bottom;}"))
        check('mobile illustration no overflow', page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1'))
        page.screenshot(path=str(OUT / 'release-mobile-light.png'), full_page=True)
        switch.click()
        check('dark moon icon visible', switch.locator('.theme-moon').is_visible() and not switch.locator('.theme-sun').is_visible())
        page.screenshot(path=str(OUT / 'release-mobile-dark.png'), full_page=True)
        switch.click()
        page.set_viewport_size({'width':1440,'height':1000})
        check("empty report", page.locator(".report-empty").is_visible())
        check("disclaimer visible in DOM", page.locator(".app-footer").count() == 1)
        check("technical accordion initially closed", not page.locator("#technical-details textarea").is_visible())
        page.screenshot(path=str(OUT / "desktop-empty.png"), full_page=True)
        page.locator("#face-upload input[type=file]").set_input_files(str(image))
        page.get_by_role("button", name="Bắt đầu phân tích", exact=True).click()
        page.locator('[data-state="processing"]').wait_for()
        check("processing has no stale cards", page.locator(".reading-card").count() == 0)
        check("duplicate submit disabled", not page.get_by_role("button", name="Bắt đầu phân tích", exact=True).is_enabled())
        page.screenshot(path=str(OUT / "desktop-processing.png"), full_page=True)
        page.locator(".reading-card").first.wait_for(timeout=30000)
        page.wait_for_function("document.querySelectorAll('.reading-card').length === 5")
        titles = page.locator(".report-title").all_text_contents()
        check("five cards in order", titles == ["Personality", "Career", "Relationships", "Future & Destiny", "Wealth"])
        check("five section navigation links", page.locator('.report-nav a').count() == 5)
        page.locator('.report-nav a').last.click()
        check("section link reaches Wealth", page.locator('#reading-section-5').evaluate("e => Math.abs(e.getBoundingClientRect().top - 24) < 3"))
        page.locator('#reading-section-5 .back-to-report').click()
        check("return to report navigation", page.locator('.report-nav').is_visible() and page.locator('#report-heading').evaluate("e => Math.abs(e.getBoundingClientRect().top - 24) < 3"))
        check("markup escaped", page.locator(".report-text script").count() == 0 and '<script>alert(1)</script>' in page.locator(".report-text").first.inner_text())
        check("no internal text scrolling", page.locator(".report-text").evaluate_all("els => els.every(e => e.scrollHeight <= e.clientHeight + 1)"))
        page.get_by_role("button", name="Sao chép toàn bộ", exact=True).click()
        page.wait_for_function("document.querySelector('.copy-feedback').textContent.includes('Đã sao chép')")
        clipboard = page.evaluate("navigator.clipboard.readText()")
        check("copy full report headings/order", all(t + ':' in clipboard for t in titles) and clipboard.index('Personality:') < clipboard.index('Wealth:'))
        page.get_by_role("button", name="Sao chép Career", exact=True).click()
        page.wait_for_function("async () => (await navigator.clipboard.readText()).startsWith('Career:')")
        section_copy = page.evaluate("navigator.clipboard.readText()").replace("\r\n", "\n")
        check("copy one section", section_copy.startswith("Career:\n"))
        check("copy preserves full text", section_copy == "Career:\n" + page.locator(".report-text").nth(1).text_content().replace("\r\n", "\n"))
        check("CSV/JSON links", page.locator('#report-downloads a[href*=".csv"]').count() >= 1 and page.locator('#report-downloads a[href*=".json"]').count() >= 1)
        with page.expect_download() as download:
            page.locator('#report-downloads a[href*=".json"]').first.click()
        saved = json.loads(Path(download.value.path()).read_text(encoding="utf-8"))
        check("JSON download original schema", len(saved["predictions"]) == 5 and saved["generator"]["method"] == "flan_t5")
        with page.expect_download() as download:
            page.locator('#report-downloads a[href*=".csv"]').first.click()
        check("CSV download original content", Path(download.value.path()).read_text(encoding="utf-8").startswith("image_id,ratio\n"))
        for width in [1440, 1024, 768, 375]:
            page.set_viewport_size({"width": width, "height": 1000})
            page.wait_for_timeout(200)
            check(f"no horizontal overflow {width}", page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1"))
            check(f"touch targets at least 44px {width}", page.locator('.copy-button, .report-nav a, .back-to-report').evaluate_all("els => els.every(e => e.getBoundingClientRect().height >= 44)"))
            if width in (1440, 375):
                page.screenshot(path=str(OUT / f"report-{width}.png"), full_page=True)
                page.locator(".reading-card").first.screenshot(path=str(OUT / f"card-{width}.png"))
        page.evaluate("document.documentElement.classList.add('dark'); document.body.classList.add('dark')")
        page.screenshot(path=str(OUT / "mobile-dark.png"), full_page=True)
        check("dark report color changes", page.locator("#physio-app").evaluate("e => getComputedStyle(e).getPropertyValue('--paper').trim() === '#1b2422'"))
        page.evaluate("document.documentElement.classList.remove('dark'); document.body.classList.remove('dark')")
        page.set_viewport_size({"width": 1440, "height": 1000})
        method = page.get_by_role("combobox").first
        method.click()
        page.get_by_role("option", name=GENERATOR_LABELS["template_0.5"], exact=True).click()
        page.locator(".report-empty").wait_for()
        check("method change clears old report", page.locator(".reading-card").count() == 0)
        page.get_by_role("button", name="Bắt đầu phân tích", exact=True).click()
        page.locator('[data-state="warning"]').wait_for(timeout=30000)
        check("warning displayed", "Mock image-quality warning" in page.locator("#analysis-status").inner_text())
        page.screenshot(path=str(OUT / "desktop-warning.png"), full_page=True)
        method.click()
        page.get_by_role("option", name=GENERATOR_LABELS["controlled_flan_guarded"], exact=True).click()
        page.get_by_role("button", name="Bắt đầu phân tích", exact=True).click()
        page.locator(".report-method").filter(has_text="fallback").wait_for(timeout=30000)
        check("fallback not presented as neural", "Mẫu xác định" in page.locator(".report-method").inner_text())
        method.click()
        page.get_by_role("option", name=GENERATOR_LABELS["controlled_deterministic"], exact=True).click()
        page.get_by_role("button", name="Bắt đầu phân tích", exact=True).click()
        page.locator('[data-state="error"]').wait_for(timeout=30000)
        check("error clears old cards", page.locator(".reading-card").count() == 0)
        check("error clears downloads", page.locator('#report-downloads a[href*=".json"]').count() == 0)
        page.screenshot(path=str(OUT / "desktop-error.png"), full_page=True)
        page.get_by_role("button", name="Xóa và bắt đầu lại", exact=True).click()
        page.get_by_text("Chưa có báo cáo", exact=True).wait_for()
        check("reset returns empty", page.locator(".reading-card").count() == 0)
        check("browser runtime errors", not errors)
        browser.close()
    report = {"scope": "Actual Gradio UI with synthetic image and mocked prediction backend; no face inference", "checks": checks, "browser_errors": errors, "console_errors": console_errors,
              "pass": sum(x["pass"] for x in checks), "fail": sum(not x["pass"] for x in checks)}
    (OUT / "browser_verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if report["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
