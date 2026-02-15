import os
from flask import Flask
from config import Config
from app.logging_config import setup_logging


def create_app():
    setup_logging(component="app")
    app = Flask(__name__)
    app.config.from_object(Config)

    os.makedirs(app.config["CACHE_FOLDER"], exist_ok=True)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Resolve S3 bucket name from SSM if not set via env var.
    # Cache the resolved value so API endpoints can read it from config.
    # Only attempt SSM lookup when a parameter name is configured (skip in
    # tests / local dev where AUDIT_S3_BUCKET_SSM_PARAM is empty).
    if not app.config.get("AUDIT_S3_BUCKET") and app.config.get("AUDIT_S3_BUCKET_SSM_PARAM"):
        from app.services.audit_logger import _resolve_bucket_name
        app.config["AUDIT_S3_BUCKET"] = _resolve_bucket_name(app.config)

    # Web UI routes
    from app.server import web as web_blueprint
    app.register_blueprint(web_blueprint)

    # REST API v1
    from app.api import api as api_blueprint
    app.register_blueprint(api_blueprint)

    return app
