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

    # Web UI routes
    from app.server import web as web_blueprint
    app.register_blueprint(web_blueprint)

    # REST API v1
    from app.api import api as api_blueprint
    app.register_blueprint(api_blueprint)

    return app
