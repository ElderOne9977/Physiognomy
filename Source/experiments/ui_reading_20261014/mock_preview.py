"""Task-owned local visual-test harness. Synthetic text/images only; no ML loading."""
import json
import sys
import tempfile
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))
import demo_gradio as D
import test_demo_gradio as T
from predict_image_to_text import PipelineError

CALLS = 0
D.load_models_once = lambda *a, **kw: 0.0


def mocked(image_path, **kw):
    global CALLS
    CALLS += 1
    print("UI_MOCK_CALL", CALLS, kw["generator"], flush=True)
    time.sleep(2)
    if kw["generator"] == "controlled_deterministic":
        raise PipelineError("FACE_NOT_DETECTED", "mock technical detail")
    result = T.fake_result(image_path)
    paragraph = ("This paragraph is a synthetic fixture for testing the reading experience. "
                 "It preserves punctuation, line breaks, and the original English wording. "
                 "A longer report should remain comfortable to read on a small screen without scrolling inside a text box. ")
    result["predictions"] = {k: paragraph * 3 + '\n\nSecond paragraph: <script>alert(1)</script> & quoted text.'
                             for k in result["predictions"]}
    result["generator"]["method"] = kw["generator"]
    if kw["generator"] == "template_0.5":
        result["face"]["warnings"] = ["Mock image-quality warning: please use a clearer image."]
    if kw["generator"] == "controlled_flan_guarded":
        result["generator"].update(neural_used=False, fallback_used=True, provenance={
            "realizer": "flan_t5_rejected_fallback", "neural_attempted": True, "fallback_used": True})
    Path(kw["features_output"]).write_text("image_id,ratio\nsynthetic,1.0\n", encoding="utf-8")
    result["features"]["csv_path"] = kw["features_output"]
    Path(kw["output"]).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    D.predict_image_to_text = mocked
    with tempfile.TemporaryDirectory(prefix="physio_ui_mock_") as td:
        D.DEMO_TMP = Path(td)
        D.build_interface().launch(server_name="127.0.0.1", server_port=7862,
                                  share=False, css_paths=[D.CSS_PATH], theme="base", footer_links=[])
