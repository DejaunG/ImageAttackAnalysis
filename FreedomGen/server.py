import subprocess
from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import sys

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

@app.route('/')
def home():
    return send_from_directory('.', 'ImageInterface.html')


# In your server.py file

@app.route('/upload', methods=['POST'])
def handle_upload():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join('uploads', filename)

    # Ensure the uploads directory exists
    os.makedirs('uploads', exist_ok=True)

    file.save(file_path)

    print("Running fgsm_adv.py script...")
    python_path = sys.executable  # Use the current Python interpreter
    result = subprocess.run([python_path, 'fgsm_adv.py', file_path],
                            capture_output=True, text=True, encoding='utf-8')

    print("Script stdout:", result.stdout)
    print("Script stderr:", result.stderr)

    # Check if images were created
    image_paths = [
        'generatedimages/adversarial_examples.png',
        'generatedimages/training_validation_plots.png',
    ]

    existing_images = [path for path in image_paths if os.path.exists(path)]

    return jsonify({
        'message': 'File successfully uploaded',
        'filename': filename,
        'script_output': result.stdout,
        'script_error': result.stderr,
        'images': existing_images
    }), 200

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)