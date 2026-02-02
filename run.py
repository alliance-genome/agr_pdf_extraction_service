from app.server import app
import os

if __name__ == '__main__':
    if not app.config['ANTHROPIC_API_KEY']:
        print('WARNING: ANTHROPIC_API_KEY environment variable not set!')
    
    print(f"Cache folder: {app.config['CACHE_FOLDER']}")
    app.run(debug=True, host='0.0.0.0', port=5000)