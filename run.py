"""
Development server entry point.

Usage:
    python run.py

For production, use gunicorn instead:
    gunicorn -w 4 -b 0.0.0.0:5000 --timeout 600 "app:create_app()"
"""

from app import create_app

app = create_app()

if __name__ == "__main__":
    if not app.config.get("ANTHROPIC_API_KEY"):
        print("WARNING: ANTHROPIC_API_KEY environment variable not set!")

    print(f"Cache folder: {app.config['CACHE_FOLDER']}")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print()
    print("API endpoints:")
    print("  GET  /api/v1/health")
    print("  POST /api/v1/extract")
    print("  GET  /api/v1/extract/<job_id>")
    print()
    app.run(debug=True, host="0.0.0.0", port=5000)
