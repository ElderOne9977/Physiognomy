"""Presentation-only regressions; explicit text fixtures, no model initialization."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

sys.path.insert(0, str(Path(__file__).parent))
import demo_gradio as D


class ReadingViewTests(unittest.TestCase):
    def output(self, text="English text.\nSecond paragraph."):
        return (text, "Career text.", "Relationship text.", "Future text.", "Wealth text.", None, None, "", "Completed in 1s.")

    def test_order_and_full_text(self):
        report, status = D.render_reading_report(self.output())
        titles = ["Personality", "Career", "Relationships", "Future &amp; Destiny", "Wealth"]
        self.assertEqual(report.count('<article '), 5)
        self.assertEqual(sorted(report.index(t) for t in titles), [report.index(t) for t in titles])
        self.assertIn("English text.\nSecond paragraph.", report)
        self.assertIn("Tính cách", report)

    def test_escape_output(self):
        report, _ = D.render_reading_report(self.output('<script>alert(1)</script><img src=x onerror=alert(2)> & "quote"'))
        self.assertNotIn("<script>", report)
        self.assertNotIn("<img src=x", report)
        self.assertIn("&lt;script&gt;", report)
        self.assertIn("&amp;", report)

    def test_long_text_not_truncated(self):
        text = "A long English sentence. " * 600
        report, _ = D.render_reading_report(self.output(text))
        self.assertIn(text, report)

    def test_missing_section_not_invented(self):
        values = list(self.output()); values[2] = ""
        report, _ = D.render_reading_report(tuple(values))
        self.assertIn("Chưa có nội dung cho phần này.", report)
        self.assertEqual(report.count('data-copy="section"'), 4)

    def test_malformed_response(self):
        values = list(self.output()); values[0] = {"bad": "shape"}
        report, status = D.render_reading_report(tuple(values))
        self.assertNotIn("<article ", report)
        self.assertIn('data-state="error"', status)

    def test_initial_empty(self):
        report, status = D.render_reading_report(D._empty())
        self.assertNotIn("<article ", report)
        self.assertIn("Chưa có báo cáo", report)

    def test_error_not_stale(self):
        values = ("", "", "", "", "", None, None, "code=FACE_NOT_DETECTED\ndetail=private", "backend error")
        report, status = D.render_reading_report(values)
        self.assertNotIn("<article ", report)
        self.assertIn("khuôn mặt", status)
        self.assertNotIn("private", status)

    def test_warning_visible(self):
        result = {"face": {"warnings": ["Quality warning"]}, "timing": {"total_s": 1}}
        _, status = D.render_reading_report(self.output(), result=result)
        self.assertIn('data-state="warning"', status)
        self.assertIn("Quality warning", status)

    def test_fallback_metadata(self):
        result = {"generator": {"method": "controlled_flan_guarded", "provenance": {
            "realizer": "flan_t5_rejected_fallback", "fallback_used": True}}, "timing": {"total_s": 2}}
        report, _ = D.render_reading_report(self.output(), generator="controlled_flan_guarded", result=result)
        self.assertIn("Mẫu xác định", report)
        self.assertIn("fallback", report)
        self.assertNotIn("Đã sinh bằng FLAN", report)

    def test_copy_order_and_no_innerhtml_assignment(self):
        report, _ = D.render_reading_report(self.output())
        self.assertEqual(report.count('data-copy="section"'), 5)
        self.assertIn('data-copy="report"', report)
        self.assertIn("navigator.clipboard.writeText", D.COPY_JS)
        self.assertNotIn("innerHTML", D.COPY_JS)

    def test_adapter_single_call_and_downloads(self):
        values = list(self.output()); values[5:7] = ["fixture.csv", "fixture.json"]
        with patch.object(D, "analyze_image", return_value=tuple(values)) as call:
            outputs = D.analyze_reading_view("fixture", D.DEFAULT_GENERATOR)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(outputs[2:4], ("fixture.csv", "fixture.json"))

    def test_adapter_failure_clears_files(self):
        with patch.object(D, "analyze_image", return_value=("", "", "", "", "", None, None, "code=INVALID_IMAGE", "error")):
            outputs = D.analyze_reading_view("fixture", D.DEFAULT_GENERATOR)
        self.assertEqual(outputs[2:4], (None, None))
        self.assertNotIn("<article ", outputs[0])

    def test_legacy_labels_preserved(self):
        for key, label in D.LEGACY_GENERATOR_LABELS.items():
            self.assertEqual(D._LABEL_TO_GENERATOR[label], key)
            self.assertEqual(D._LABEL_TO_GENERATOR[D.GENERATOR_LABELS[key]], key)

    def test_malformed_metadata_does_not_break_reading(self):
        report, status = D.render_reading_report(self.output(), result={"generator": [], "face": [], "timing": []})
        self.assertEqual(report.count("<article "), 5)

    def test_deterministic_baseline_not_called_neural_fallback(self):
        result = {"generator": {"method": "controlled_deterministic", "provenance": {"realizer": "deterministic", "fallback_used": True}}}
        report, _ = D.render_reading_report(self.output(), result=result)
        self.assertIn("Mẫu xác định (controlled)", report)
        self.assertNotIn("neural bị từ chối", report)

    def test_progress_scoped_to_image(self):
        import gradio as gr
        demo = D.build_interface()
        image = next(c for c in demo.blocks.values() if isinstance(c, gr.Image))
        event = next(d for d in demo.config["dependencies"]
                     if demo.fns[d["id"]].fn.__name__ == "_analyze")
        self.assertEqual(event["show_progress_on"], [image._id])
        self.assertEqual(event["show_progress"], "full")
        for other in demo.config["dependencies"]:
            if other["id"] != event["id"]:
                self.assertEqual(other["show_progress"], "hidden")


if __name__ == "__main__":
    unittest.main()
