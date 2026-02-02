import os
from config import Config
from flask import Flask

def create_app():
  app = Flask(__name__)
  app.config.from_object(Config)

  os.makedirs(app.config['CACHE_FOLDER'], exist_ok=True)
  os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

  with app.app_context():
      from app import server

  return app
