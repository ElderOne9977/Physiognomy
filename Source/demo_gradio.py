# -*- coding: utf-8 -*-
"""
Thin Gradio wrapper around predict_image_to_text().
Launch: python Source/demo_gradio.py
Local only (share=False). Models loaded once, reused across requests.
"""
from __future__ import annotations
import argparse
import html
import json
import os
import sys
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from predict_image_to_text import (  # noqa: E402
    GENERATOR_CHOICES,
    PipelineError,
    TEMPLATE_THRESHOLDS_PATH,
    TEXT_MODEL_DEFAULT,
    extract_features_from_image,
    friendly_message,
    generate_text_from_semantics,
    load_semantic_artifacts,
    load_text_generator,
    predict_image_to_text,
    predict_semantic_from_dataframe,
)

DEMO_TMP = ROOT / "outputs" / "demo_tmp"
MAX_UPLOAD_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
MAX_IMAGE_PIXELS = 4096 * 4096            # ~16.7 megapixels
DEMO_FILE_TTL_SECONDS = 7200              # 2 hours
DISCLOSURE = (
    "Experimental demonstration. Generated interpretations imitate a "
    "face-reading text provider and are not scientifically validated "
    "assessments of a person."
)
SUPPORTED = [".jpg", ".jpeg", ".png", ".webp"]
GENERATOR_LABELS = {
    "flan_t5": "FLAN-T5 (continuous, experimental)",
    "template_0.5": "Template @ 0.5",
    "template_calibrated": "Template (calibrated, recommended)",
    "template_concise": "Template Concise",
    "controlled_deterministic": "Controlled (planner + template)",
    "controlled_flan_guarded": "Controlled (planner + FLAN-T5, guarded)",
}
LEGACY_GENERATOR_LABELS = GENERATOR_LABELS.copy()
GENERATOR_LABELS = {
    "flan_t5": "FLAN-T5 · Neural (experimental)",
    "template_0.5": "Template 0.5 · Mẫu cố định",
    "template_calibrated": "Template Calibrated · Chi tiết (recommended)",
    "template_concise": "Template Concise · Ngắn gọn",
    "controlled_deterministic": "Controlled · Nội dung có cấu trúc",
    "controlled_flan_guarded": "Controlled FLAN · Có kiểm tra (experimental)",
}
GENERATOR_HELP = "FLAN-T5 là thử nghiệm. Chọn Template Calibrated nếu ưu tiên tính nhất quán giữa đầu vào và văn bản."
CSS_PATH = SRC / "assets" / "demo.css"
COPY_JS = (SRC / "assets" / "demo-copy.js").read_text(encoding="utf-8")
SECTIONS = (
    ("Personality", "Tính cách", '<circle cx="12" cy="8" r="3"/><path d="M6 20v-2a6 6 0 0 1 12 0v2"/>'),
    ("Career", "Sự nghiệp", '<rect x="4" y="8" width="16" height="12" rx="2"/><path d="M8 8V4h8v4M4 13h16"/>'),
    ("Relationships", "Mối quan hệ", '<circle cx="8" cy="8" r="3"/><circle cx="17" cy="9" r="2"/><path d="M2 20v-2a6 6 0 0 1 12 0v2m1-7a5 5 0 0 1 7 5v2"/>'),
    ("Future & Destiny", "Tương lai", '<path d="M3 17h18M12 2v3M4 6l2 2m14-2-2 2"/><path d="M6 17a6 6 0 0 1 12 0"/>'),
    ("Wealth", "Tài chính", '<rect x="3" y="6" width="18" height="14" rx="2"/><path d="M3 10h18m-5 4h5M6 6V3h11"/>'),
)
GENERATOR_VALUES = list(GENERATOR_CHOICES)

def get_default_generator() -> str:
    """Return default generator from PHYSIO_DEFAULT_GENERATOR env var, falling back safely to flan_t5."""
    val = os.environ.get("PHYSIO_DEFAULT_GENERATOR", "flan_t5").strip()
    return val if val in GENERATOR_VALUES else "flan_t5"

DEFAULT_GENERATOR = get_default_generator()
# ponytail: 3 generators cover collapse diagnosis; add learned projection only after template vs FLAN gap measured
_LABEL_TO_GENERATOR = {label: key for labels in (LEGACY_GENERATOR_LABELS, GENERATOR_LABELS) for key, label in labels.items()}
_lock = threading.Lock()
_models_ready = False
_init_s = None


def _ensure_tmp():
    DEMO_TMP.mkdir(parents=True, exist_ok=True)


def load_models_once(generator=None):
    """Load SCRFD + semantic + (FLAN-T5 if needed) once. Safe to call repeatedly."""
    global _models_ready, _init_s
    if _models_ready:
        return _init_s or 0.0
    if generator is None:
        generator = get_default_generator()
    t0 = time.time()
    load_semantic_artifacts()
    if generator in ("flan_t5", "controlled_flan_guarded"):
        ctrl_path = SRC / "models_text" / "flan_t5_small_controlled"
        target_path = ctrl_path if (generator == "controlled_flan_guarded" and ctrl_path.exists()) else TEXT_MODEL_DEFAULT
        load_text_generator(str(target_path))
    else:
        if generator in ("template_calibrated", "template_concise") and not TEMPLATE_THRESHOLDS_PATH.exists():
            raise PipelineError("MODEL_MISSING", f"calibrated thresholds not found: {TEMPLATE_THRESHOLDS_PATH} (no silent fallback to 0.5)")
        if generator in ("controlled_deterministic", "controlled_flan_guarded") and not TEMPLATE_THRESHOLDS_PATH.exists():
            raise PipelineError("MODEL_MISSING", f"calibrated thresholds not found: {TEMPLATE_THRESHOLDS_PATH} (no silent fallback to 0.5)")
    try:
        from physiognomy import ensure_heavy_models
        ensure_heavy_models()
    except Exception:
        # detector loads lazily on first extract; do not fail UI startup
        pass
    _init_s = round(time.time() - t0, 3)
    _models_ready = True
    return _init_s


def warmup_models():
    return load_models_once()


def cleanup_old_temp_files(max_age_seconds: int = DEMO_FILE_TTL_SECONDS) -> int:
    """Remove demo_tmp artifacts older than max_age_seconds. Returns count of removed files."""
    if not DEMO_TMP.exists():
        return 0
    now = time.time()
    removed = 0
    for p in DEMO_TMP.glob("*"):
        if p.is_file():
            try:
                mtime = p.stat().st_mtime
                if now - mtime > max_age_seconds:
                    p.unlink()
                    removed += 1
            except OSError:
                pass
    return removed


def check_readiness() -> dict:
    """Return health/readiness status dict for Docker/k8s probes."""
    return {
        "status": "ready" if _models_ready else "loading",
        "models_ready": _models_ready,
        "init_time_s": _init_s,
        "default_generator": get_default_generator(),
        "generators": list(GENERATOR_CHOICES),
        "upload_limits": {
            "max_upload_size_bytes": MAX_UPLOAD_SIZE_BYTES,
            "max_image_pixels": MAX_IMAGE_PIXELS,
            "demo_file_ttl_seconds": DEMO_FILE_TTL_SECONDS,
        },
    }


def _save_upload(image) -> Path:
    """Persist Gradio image (path/ndarray/PIL) to a unique temp file. Cleanup: demo_tmp, session-scoped."""
    _ensure_tmp()
    try:
        cleanup_old_temp_files(DEMO_FILE_TTL_SECONDS)
    except Exception:
        pass
    uid = uuid.uuid4().hex[:12]
    dest = DEMO_TMP / f"{uid}.jpg"
    if image is None:
        raise PipelineError("FILE_NOT_FOUND", "no image uploaded")
    if isinstance(image, (str, Path)):
        src = Path(image)
        if not src.exists():
            raise PipelineError("FILE_NOT_FOUND", str(src))
        if src.suffix.lower() not in SUPPORTED:
            raise PipelineError("INVALID_IMAGE", f"unsupported format {src.suffix}")
        size = src.stat().st_size
        if size > MAX_UPLOAD_SIZE_BYTES:
            raise PipelineError("INVALID_IMAGE", f"upload size {size} bytes exceeds limit of {MAX_UPLOAD_SIZE_BYTES} bytes")
        from PIL import Image
        try:
            im = Image.open(src)
        except Exception as e:
            raise PipelineError("INVALID_IMAGE", f"cannot decode image: {e}")
        w, h = im.size
        if w * h > MAX_IMAGE_PIXELS:
            raise PipelineError("INVALID_IMAGE", f"image resolution {w}x{h} ({w*h} pixels) exceeds limit of {MAX_IMAGE_PIXELS} pixels")
        im.convert("RGB").save(dest, "JPEG", quality=95)
        return dest
    from PIL import Image as PILImage
    import numpy as np
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
        if w * h > MAX_IMAGE_PIXELS:
            raise PipelineError("INVALID_IMAGE", f"image resolution {w}x{h} ({w*h} pixels) exceeds limit of {MAX_IMAGE_PIXELS} pixels")
        im = PILImage.fromarray(image)
    elif hasattr(image, "save"):
        im = image
        w, h = getattr(im, "size", (0, 0))
        if w and h and w * h > MAX_IMAGE_PIXELS:
            raise PipelineError("INVALID_IMAGE", f"image resolution {w}x{h} ({w*h} pixels) exceeds limit of {MAX_IMAGE_PIXELS} pixels")
    else:
        raise PipelineError("INVALID_IMAGE", f"unsupported upload type {type(image)}")
    im.convert("RGB").save(dest, "JPEG", quality=95)
    return dest


def _empty():
    return "", "", "", "", "", None, None, "", "Idle."


def analyze_image(image, generator=None, progress=None):
    """UI callback. Reuses predict_image_to_text. Serializes concurrent runs."""
    if not _lock.acquire(blocking=False):
        return ("", "", "", "", "", None, None,
                "Another analysis is running. Wait for it to finish.",
                "Busy.")
    try:
        if progress is not None:
            try:
                progress(0, desc="Chuẩn bị ảnh...")
            except Exception:
                pass
        if image is None:
            return (*_empty()[:5], None, None,
                    friendly_message("FILE_NOT_FOUND"),
                    "Upload an image first.")
        if generator is None:
            generator = get_default_generator()
        gen = generator if generator in GENERATOR_VALUES else _LABEL_TO_GENERATOR.get(generator)
        if gen is None:
            raise PipelineError("TEXT_GENERATION_FAILED", f"unknown generator {generator!r}")
        load_models_once(generator=gen)
        path = _save_upload(image)
        stem = path.stem
        csv_out = DEMO_TMP / f"{stem}_features.csv"
        json_out = DEMO_TMP / f"{stem}_result.json"
        if progress is not None:
            try:
                progress(0.25, desc="Đang xử lý ảnh và tạo văn bản...")
            except Exception:
                pass
        result = predict_image_to_text(
            path,
            features_output=str(csv_out),
            output=str(json_out),
            feature_source="extract",
            generator=gen,
        )
        if progress is not None:
            try:
                progress(1.0, desc="Đã chuẩn bị báo cáo.")
            except Exception:
                pass
        pred = result.get("predictions") or {}
        face = result.get("face") or {}
        warns = face.get("warnings") or []
        status_bits = []
        if face.get("multiple_faces_detected") or any("MULTIPLE_FACES" in str(w) for w in warns):
            status_bits.append(friendly_message("MULTIPLE_FACES_WARNING"))
        qc = face.get("qc_reasons")
        if qc and qc != "ok":
            status_bits.append(f"Image quality: {qc.replace('|', ', ')}.")
        for w in warns:
            ws = str(w)
            if "MULTIPLE_FACES" not in ws and ws not in status_bits:
                status_bits.append(ws)
        timing = result.get("timing") or {}
        status_bits.append(f"Completed in {timing.get('total_s', '?')}s.")
        tech = _format_tech(result)
        csv_path = result.get("features", {}).get("csv_path")
        json_path = str(json_out) if json_out.exists() else None
        return (
            pred.get("personality") or "",
            pred.get("career") or "",
            pred.get("relationships") or "",
            pred.get("future_destiny") or "",
            pred.get("wealth") or "",
            csv_path if csv_path and Path(csv_path).exists() else None,
            json_path,
            tech,
            " ".join(status_bits),
        )
    except PipelineError as e:
        msg = friendly_message(e.code, e.message)
        tech = f"code={e.code}\ndetail={e.message}"
        return ("", "", "", "", "", None, None, tech, msg)
    except FileNotFoundError as e:
        return ("", "", "", "", "", None, None, "", friendly_message("FILE_NOT_FOUND"))
    except Exception as e:
        tech = traceback.format_exc()
        return ("", "", "", "", "", None, None, tech, friendly_message("FEATURE_EXTRACTION_FAILED"))
    finally:
        _lock.release()


def _format_tech(result: dict) -> str:
    face = result.get("face") or {}
    feats = result.get("features") or {}
    sem = result.get("semantic") or {}
    timing = result.get("timing") or {}
    probs = sem.get("probabilities") or {}
    gen_info = result.get("generator") or {}
    lines = [
        f"face detected: {face.get('detected')}",
        f"num faces / multi: {face.get('multiple_faces_detected')}",
        f"selected face: {face.get('selected_face')}",
        f"detection score: {face.get('detection_score')}",
        f"qc: {face.get('qc_reasons', face.get('qc_pass'))}",
        f"warnings: {face.get('warnings')}",
        f"feature source: {result.get('feature_source')}",
        f"raw feature count: {feats.get('raw_feature_count')}",
        f"semantic trait count: {sem.get('trait_count')}",
        f"generator: {gen_info.get('method') or result.get('model')}",
        f"generator model: {gen_info.get('model') or result.get('model')}",
        f"generator representation: {gen_info.get('representation') or 'continuous'}",
        f"threshold policy: {gen_info.get('threshold_policy')}",
        f"timing: {timing}",
    ]
    if gen_info.get("provenance") is not None:
        prov = gen_info.get("provenance") or {}
        lines.extend([
            f"controlled realizer: {prov.get('realizer')}",
            f"neural generation used: {gen_info.get('neural_used')}",
            f"deterministic fallback used: {gen_info.get('fallback_used')}",
            f"neural attempted: {prov.get('neural_attempted')}",
        ])
        if prov.get("rejection_reasons"):
            lines.append(f"validator rejections: {prov.get('rejection_reasons')}")
        cp = gen_info.get("content_plan") or {}
        if cp.get("selected_all") is not None:
            lines.append(f"plan selected: {cp.get('selected_all')}")
            lines.append(f"plan omitted: {cp.get('omitted_all')}")
    lines.extend(["", "semantic probabilities:"])
    for k, v in sorted(probs.items()):
        try:
            lines.append(f"  {k}: {float(v):.4f}")
        except Exception:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def _notice(message, state="empty"):
    return f'<div class="report-notice" data-state="{state}" role="status" aria-live="polite">{html.escape(message)}</div>'


def _report_empty(title="Chưa có báo cáo", body="Chọn một ảnh và bắt đầu phân tích. Kết quả sẽ xuất hiện tại đây."):
    return f'<div class="report-empty"><h2>{html.escape(title)}</h2><p>{html.escape(body)}</p></div>'


def render_reading_report(values, generator=None, result=None):
    """Escape and display existing callback text; never infer or rewrite content."""
    if generator is None:
        generator = get_default_generator()
    if not isinstance(values, (tuple, list)) or len(values) != 9 or not all(isinstance(v, str) for v in values[:5]):
        return _report_empty("Không thể hiển thị báo cáo", "Định dạng kết quả không hợp lệ. Hãy thử lại."), _notice("Định dạng kết quả không hợp lệ.", "error")
    result = result if isinstance(result, dict) else {}
    if not any(values[:5]):
        if values[8] == "Idle.":
            return _report_empty(), _notice("Sẵn sàng. Chọn ảnh để bắt đầu.")
        messages = {
            "FACE_NOT_DETECTED": "Không tìm thấy khuôn mặt. Hãy thử ảnh rõ nét hơn.",
            "FILE_NOT_FOUND": "Không tìm thấy ảnh. Hãy chọn lại ảnh.",
            "INVALID_IMAGE": "Ảnh không hợp lệ. Hãy thử JPG, JPEG, PNG hoặc WEBP.",
            "MODEL_MISSING": "Thiếu tài nguyên mô hình. Xem chi tiết kỹ thuật để kiểm tra.",
            "TEXT_GENERATION_FAILED": "Không thể tạo báo cáo. Kiểm tra phương thức đã chọn hoặc thử lại.",
        }
        code = next((c for c in messages if f"code={c}" in values[7]), None)
        message = messages.get(code, "Không thể hoàn tất phân tích. Hãy thử lại hoặc xem chi tiết kỹ thuật.")
        if values[8] == "Upload an image first.":
            message = "Hãy chọn một ảnh trước khi phân tích."
        if values[8] == "Busy.":
            message = "Một yêu cầu đang được xử lý. Vui lòng đợi hoàn tất."
        return _report_empty("Chưa thể tạo báo cáo", message), _notice(message, "error")
    gen = result.get("generator") if isinstance(result.get("generator"), dict) else {}
    actual = gen.get("method") or (generator if generator in GENERATOR_VALUES else _LABEL_TO_GENERATOR.get(generator))
    method = GENERATOR_LABELS.get(actual, "Chưa có metadata phương thức")
    provenance = gen.get("provenance") if isinstance(gen.get("provenance"), dict) else {}
    if provenance.get("realizer") == "flan_t5_rejected_fallback":
        method = "Mẫu xác định (fallback; đầu ra neural bị từ chối)"
    elif provenance.get("realizer") == "deterministic":
        method = "Mẫu xác định (controlled)"
    elif actual == "controlled_flan_guarded" and not provenance:
        method = "Controlled FLAN; chưa có metadata phương thức cuối"
    cards = []
    for i, ((title, subtitle, icon), text) in enumerate(zip(SECTIONS, values[:5]), 1):
        content = html.escape(text) if text else "Chưa có nội dung cho phần này."
        copy = f'<button type="button" class="copy-button" data-copy="section" aria-label="Sao chép {title}">Sao chép</button>' if text else ""
        cards.append(f'<article class="reading-card" id="reading-section-{i}" tabindex="-1"><header><span class="category-icon" aria-hidden="true"><svg viewBox="0 0 24 24">{icon}</svg></span><div><h3 class="report-title">{html.escape(title)}</h3><p class="category-subtitle">{i:02d} · {subtitle}</p></div>{copy}</header><p class="report-text" data-available="{str(bool(text)).lower()}">{content}</p><a class="back-to-report" href="#report-heading">Về mục lục ↑</a></article>')
    navigation = '<nav class="report-nav" aria-label="Mục lục báo cáo">' + ''.join(
        f'<a href="#reading-section-{i}"><span>{i:02d}</span> {html.escape(subtitle)}</a>'
        for i, (_, subtitle, _) in enumerate(SECTIONS, 1)) + '</nav>'
    report = ('<section class="reading-report" aria-label="Generated report">'
              '<div class="report-heading" id="report-heading" tabindex="-1"><div><p class="eyebrow">Báo cáo · 5 phần</p><h2>Kết quả phân tích</h2>'
              f'<p class="report-method">{html.escape(method)}</p></div><button type="button" class="copy-button copy-all" data-copy="report">Sao chép toàn bộ</button></div>'
              + navigation + "".join(cards) + '<p class="copy-feedback" role="status" aria-live="polite"></p></section>')
    face = result.get("face") if isinstance(result.get("face"), dict) else {}
    warnings = face.get("warnings") or []
    if face.get("multiple_faces_detected") and not warnings:
        warnings = ["Ảnh có nhiều khuôn mặt; hệ thống chọn khuôn mặt lớn nhất."]
    if face.get("qc_reasons") and face["qc_reasons"] != "ok":
        warnings = [*warnings, f"Chất lượng ảnh: {face['qc_reasons']}"]
    timing = result.get("timing") if isinstance(result.get("timing"), dict) else {}
    elapsed = timing.get("total_s")
    message = "Hoàn tất." + (f" Thời gian: {elapsed}s." if isinstance(elapsed, (int, float)) else "")
    if warnings:
        message += " " + " ".join(str(w) for w in warnings)
    return report, _notice(message, "warning" if warnings else "success")


def analyze_reading_view(image, generator=None, progress=None):
    """Presentation adapter; preserve analyze_image's nine-value public contract."""
    if generator is None:
        generator = get_default_generator()
    values = analyze_image(image, generator, progress)
    result = None
    if values[6]:
        try:
            result = json.loads(Path(values[6]).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    report, status = render_reading_report(values, generator, result)
    malformed = not all(isinstance(v, str) for v in values[:5])
    return report, status, None if malformed else values[5], None if malformed else values[6], values[7]


def build_interface():
    import gradio as gr
    default_gen = get_default_generator()
    initial_report, initial_status = render_reading_report(_empty(), generator=default_gen)
    with gr.Blocks(title="Physiognomy", analytics_enabled=False) as demo:
      with gr.Column(elem_id="physio-app"):
        gr.HTML('<header class="app-header"><div><span class="app-mark" aria-hidden="true">P</span><h1>Physiognomy</h1></div><span class="theme-control"><button type="button" role="switch" aria-checked="false" aria-label="Chế độ tối"><span class="theme-thumb"><svg class="theme-sun" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5"/></svg><svg class="theme-moon" viewBox="0 0 24 24" aria-hidden="true"><path d="M20 15A9 9 0 0 1 9 4a9 9 0 1 0 11 11Z"/></svg></span></button></span></header><p class="app-subtitle">Khám phá dữ liệu hình học khuôn mặt.</p>', js_on_load=(SRC / "assets" / "theme.js").read_text(encoding="utf-8"))
        gr.HTML((SRC / "assets" / "feature-guide.html").read_text(encoding="utf-8"))
        with gr.Row(elem_id="input-area"):
            with gr.Column(scale=1, min_width=260):
                image = gr.Image(
                    label="01 · Chọn hoặc kéo thả ảnh",
                    type="filepath",
                    sources=["upload"],
                    height=260,
                    elem_id="face-upload",
                )
                gr.Markdown("Chọn ảnh rõ nét, có thể nhìn thấy khuôn mặt. JPG · JPEG · PNG · WEBP", elem_id="upload-help")
            with gr.Column(scale=1, min_width=260, elem_id="analysis-controls"):
                gr.Markdown("### 02 · Tạo báo cáo\nKết quả gồm 5 phần, có thể sao chép hoặc tải xuống.")
                method = gr.Dropdown(
                    label="Text Generation Method",
                    choices=[GENERATOR_LABELS[v] for v in GENERATOR_VALUES],
                    value=GENERATOR_LABELS[default_gen],
                    info=GENERATOR_HELP,
                )
                with gr.Row():
                    analyze_btn = gr.Button("Bắt đầu phân tích", variant="primary", elem_id="analyze-button")
                    clear_btn = gr.Button("Xóa và bắt đầu lại", elem_id="clear-button")
        status = gr.HTML(initial_status, elem_id="analysis-status")
        report = gr.HTML(initial_report, elem_id="reading-view", js_on_load=COPY_JS)
        with gr.Row(elem_id="report-downloads"):
            csv_file = gr.File(label="Tải đặc trưng CSV", interactive=False, visible=False)
            json_file = gr.File(label="Tải kết quả JSON", interactive=False, visible=False)
        with gr.Accordion("Technical Details", open=False, elem_id="technical-details"):
            tech = gr.Textbox(label="Thông tin kỹ thuật", lines=12, interactive=False, buttons=["copy"])
        gr.HTML(f'<footer class="app-footer"><strong>Lưu ý về nội dung</strong><p>{html.escape(DISCLOSURE)}</p><p>Các diễn giải không phải đánh giá đã được kiểm chứng về một người.</p></footer>')
        outs = [report, status, csv_file, json_file, tech]
        controls = [analyze_btn, clear_btn, method, image]
        def _start():
            return (_report_empty("Đang xử lý ảnh", "Vui lòng đợi. Báo cáo mới sẽ thay thế kết quả trước."), _notice("Đang chuẩn bị ảnh và tạo báo cáo…", "processing"), gr.update(value=None, visible=False), gr.update(value=None, visible=False), "", *[gr.update(interactive=False) for _ in controls])
        def _analyze(img, gen, progress=gr.Progress()):
            report_value, status_value, csv, result_json, details = analyze_reading_view(img, gen, progress)
            return (report_value, status_value, gr.update(value=csv, visible=bool(csv)), gr.update(value=result_json, visible=bool(result_json)), details, *[gr.update(interactive=True) for _ in controls])
        analyze_btn.click(_start, outputs=[*outs, *controls], queue=False, trigger_mode="once", show_progress="hidden").then(
            _analyze, inputs=[image, method], outputs=[*outs, *controls], concurrency_limit=1,
            show_progress="full", show_progress_on=image)
        def _clear():
            cur_default = get_default_generator()
            return GENERATOR_LABELS[cur_default], None, initial_report, initial_status, gr.update(value=None, visible=False), gr.update(value=None, visible=False), ""
        clear_btn.click(_clear, outputs=[method, image, *outs], queue=False, show_progress="hidden")
        def _reset_report():
            return initial_report, initial_status, gr.update(value=None, visible=False), gr.update(value=None, visible=False), ""
        image.upload(_reset_report, outputs=outs, queue=False, show_progress="hidden")
        image.clear(_reset_report, outputs=outs, queue=False, show_progress="hidden")
        method.change(_reset_report, outputs=outs, queue=False, show_progress="hidden")
    return demo


def main():
    ap = argparse.ArgumentParser(description="Local Gradio demo for image→CSV→text")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--check-health", action="store_true", help="check readiness status and exit")
    ap.add_argument("--warmup", action="store_true", help="warm up models before running or checking health")
    args = ap.parse_args()
    if args.warmup:
        load_models_once()
    if args.check_health:
        ready = check_readiness()
        print(json.dumps(ready, indent=2))
        sys.exit(0 if ready["models_ready"] else 1)
    print(f"Loading models (CPU, first start can take ~30s)...")
    t = load_models_once()
    print(f"Models ready in {t}s. Starting local server (share=False).")
    demo = build_interface()
    try:
        demo.launch(server_name=args.host, server_port=args.port, share=False, show_error=args.debug, css_paths=[CSS_PATH], theme="base")
    except OSError as e:
        print(f"Could not bind {args.host}:{args.port}: {e}", file=sys.stderr)
        print("Try --port 7861", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
