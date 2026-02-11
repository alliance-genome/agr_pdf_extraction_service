import os
import logging
from flask import Flask
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    os.makedirs(app.config["CACHE_FOLDER"], exist_ok=True)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    with app.app_context():
        # Web UI routes (existing)
        from app import server  # noqa: F401

        # REST API v1
        from app.api import api as api_blueprint
        app.register_blueprint(api_blueprint)

    return app
