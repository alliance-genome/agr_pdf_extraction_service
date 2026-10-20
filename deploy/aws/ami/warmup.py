"""Exercise the bake's actual worker, including lazy OCR, without LLM calls.

Run inside pdfx-app, with the Compose result backend and fixture files available.
The aggregate HTTP status is insufficient: partial extractor success is allowed
by the application, but must never qualify a prewarmed AMI.
"""

import math
import os
from pathlib import Path


def positive_seconds(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def validate_result(result: dict, methods: tuple[str, ...], *, require_ocr=False):
    if not isinstance(result, dict):
        raise ValueError("Warm-up returned no structured extractor result")
    if (
        result.get("status") != "success"
        or result.get("extraction_status") != "complete_success"
        or set(result.get("available_extractors", [])) != set(methods)
        or result.get("failed_extractors")
        or result.get("extractor_failures")
        or result.get("cached_methods")
    ):
        raise ValueError(
            "Warm-up requires fresh success from every requested extractor; "
            f"available={result.get('available_extractors')}, "
            f"failed={result.get('failed_extractors')}, "
            f"cached={result.get('cached_methods')}"
        )
    extractions = result.get("extractions", {})
    for method in methods:
        text = extractions.get(method)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Warm-up {method} output is empty")
    if require_ocr:
        # This phrase is near the start, within the Celery result's preview.
        # The scan has no text layer; recovering it proves OCR actually ran.
        text = " ".join(extractions["docling"].lower().split())
        if "this paragraph exercises" not in text:
            raise ValueError("Scanned warm-up did not recover the expected OCR text")


def run_warmups(session, celery, fixtures: Path, *, job_timeout, http_timeout):
    cases = (
        ("test-sample.pdf", ("grobid", "docling", "marker"), False),
        ("test-scan.pdf", ("docling",), True),
    )
    for filename, methods, require_ocr in cases:
        with (fixtures / filename).open("rb") as pdf:
            response = session.post(
                "http://127.0.0.1:5000/api/v1/extract",
                files={"file": (filename, pdf, "application/pdf")},
                data={
                    "methods": ",".join(methods),
                    "merge": "false",
                    "extract_images": "false",
                    "review_images": "false",
                    "clear_cache_scope": "extraction",
                },
                timeout=http_timeout,
            )
        response.raise_for_status()
        process_id = response.json()["process_id"]
        print(f"Warm-up {filename}: process_id={process_id}", flush=True)
        result = celery.AsyncResult(process_id).get(timeout=job_timeout)
        validate_result(result, methods, require_ocr=require_ocr)
        print(f"Warm-up {filename}: all requested extractors passed", flush=True)


def main():
    import requests
    from celery import Celery

    job_timeout = positive_seconds("PDFX_BAKE_WARMUP_TIMEOUT_SECONDS", 600)
    http_timeout = positive_seconds("PDFX_BAKE_HTTP_TIMEOUT_SECONDS", 30)
    celery = Celery(backend=os.environ["CELERY_RESULT_BACKEND"])
    with requests.Session() as session:
        run_warmups(
            session,
            celery,
            Path(__file__).parent,
            job_timeout=job_timeout,
            http_timeout=http_timeout,
        )


if __name__ == "__main__":
    main()
