"""Browser and API-documentation routes for PDFX.

The browser submits work through ``/api/v1/extract``. Extraction and merge
execution intentionally live only in the queued Celery path.
"""

import os

from flask import Blueprint, current_app, render_template, send_file, url_for


web = Blueprint("web", __name__)


@web.route("/")
def index():
    return render_template("index.html")


@web.route("/openapi.yaml")
def openapi_spec():
    """Serve the OpenAPI specification for the REST API."""

    spec_path = os.path.join(current_app.root_path, "openapi.yaml")
    return send_file(spec_path, mimetype="application/yaml")


@web.route("/docs")
def swagger_docs():
    """Serve Swagger UI backed by the local OpenAPI spec."""

    return render_template("swagger.html", spec_url=url_for("web.openapi_spec"))
