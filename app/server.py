"""
Web UI routes for the PDF Extraction Service.

These serve the original browser-based interface (upload form + synchronous
processing).  The REST API lives in app/api.py.
"""

import os
from flask import current_app, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename

from app.services.grobid_service import Grobid
from app.services.docling_service import Docling
from app.services.marker_service import Marker
from app.services.llm_service import LLM
from app.utils import allowed_file, get_file_hash, get_cached_path, is_extraction_cached


@current_app.route("/")
def index():
    return render_template("index.html")


@current_app.route("/process", methods=["POST"])
def process_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    selected_methods = request.form.getlist("methods")
    merge_enabled = request.form.get("merge") == "on"

    if not selected_methods:
        return jsonify({"error": "Please select at least one extraction method"}), 400

    try:
        cfg = current_app.config
        filename = secure_filename(file.filename)
        pdf_path = os.path.join(cfg["UPLOAD_FOLDER"], filename)
        file.save(pdf_path)

        file_hash = get_file_hash(pdf_path)

        cached_methods = []
        extractions = {}
        methods_used = []

        if "grobid" in selected_methods:
            grobid_output = get_cached_path(file_hash, "grobid")
            if is_extraction_cached(file_hash, "grobid"):
                cached_methods.append("GROBID")
            else:
                grobid = Grobid(
                    base_url=cfg["GROBID_URL"],
                    timeout=cfg["GROBID_REQUEST_TIMEOUT"],
                    include_coordinates=cfg["GROBID_INCLUDE_COORDINATES"],
                    include_raw_citations=cfg["GROBID_INCLUDE_RAW_CITATIONS"],
                )
                grobid.extract(pdf_path, grobid_output)
            with open(grobid_output, "r", encoding="utf-8") as f:
                extractions["grobid"] = f.read()
            methods_used.append("GROBID")

        if "docling" in selected_methods:
            docling_output = get_cached_path(file_hash, "docling")
            if is_extraction_cached(file_hash, "docling"):
                cached_methods.append("Docling")
            else:
                docling = Docling(device=cfg["DOCLING_DEVICE"])
                docling.extract(pdf_path, docling_output)
            with open(docling_output, "r", encoding="utf-8") as f:
                extractions["docling"] = f.read()
            methods_used.append("Docling")

        if "marker" in selected_methods:
            marker_output = get_cached_path(file_hash, "marker")
            if is_extraction_cached(file_hash, "marker"):
                cached_methods.append("Marker")
            else:
                marker = Marker(
                    device=cfg["MARKER_DEVICE"],
                    extract_images=cfg["MARKER_EXTRACT_IMAGES"],
                )
                marker.extract(pdf_path, marker_output)
            with open(marker_output, "r", encoding="utf-8") as f:
                extractions["marker"] = f.read()
            methods_used.append("Marker")

        response_data = {
            "status": "success",
            "file_hash": file_hash,
            "cached_methods": cached_methods,
            "methods_used": methods_used,
        }

        if merge_enabled and len(selected_methods) >= 1:
            version = cfg["EXTRACTION_CONFIG_VERSION"]
            cache_key = f"v{version}_{file_hash}_{'_'.join(sorted(selected_methods))}"
            merged_cache_path = os.path.join(cfg["CACHE_FOLDER"], f"{cache_key}_merged.md")

            if os.path.exists(merged_cache_path):
                with open(merged_cache_path, "r", encoding="utf-8") as f:
                    merged_md = f.read()
                cached_methods.append("Merged")
            else:
                llm = LLM(
                    api_key=cfg["ANTHROPIC_API_KEY"],
                    model=cfg["LLM_MODEL"],
                    max_tokens=cfg["LLM_MAX_TOKENS"],
                )
                merged_md = llm.extract(
                    extractions.get("grobid", ""),
                    extractions.get("docling", ""),
                    extractions.get("marker", ""),
                )
                with open(merged_cache_path, "w", encoding="utf-8") as f:
                    f.write(merged_md)
                merged_output_path = get_cached_path(file_hash, "merged")
                with open(merged_output_path, "w", encoding="utf-8") as f:
                    f.write(merged_md)

            response_data["merged_output"] = merged_md
        else:
            response_data["individual_outputs"] = {
                method.upper(): content for method, content in extractions.items()
            }

        # Cleanup temporary PDF file (keep cached extractions)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@current_app.route("/download/<file_hash>/<method>")
def download_extraction(file_hash, method):
    try:
        if method not in ["grobid", "docling", "marker", "merged"]:
            return jsonify({"error": "Invalid method"}), 400

        filepath = get_cached_path(file_hash, method)
        if not os.path.exists(filepath):
            return jsonify({"error": "File not found"}), 404

        return send_file(
            filepath,
            as_attachment=True,
            download_name=f"{method}_extraction.md",
            mimetype="text/markdown",
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
