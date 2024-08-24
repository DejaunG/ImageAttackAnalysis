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

@app.route('/styles.css')
def styles():
    return send_from_directory('.', 'styles.css')
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
    train_dir = 'Fish-DataSets/fish/train'
    val_dir = 'Fish-DataSets/fish/val'
    result = subprocess.run([python_path, 'fgsm_adv.py', train_dir, val_dir, file_path],
                            capture_output=True, text=True, encoding='utf-8')

    print("Script stdout:", result.stdout)
    print("Script stderr:", result.stderr)

    # Check if images were created
    image_paths = [
        'generatedimages/adversarial_examples.png',
        'generatedimages/training_history.png',
    ]

    existing_images = ['uploads/' + filename] + [path for path in image_paths if os.path.exists(path)]

    if result.returncode != 0:
        return jsonify({
            'error': 'Script failed',
            'script_output': result.stdout,
            'script_error': result.stderr
        }), 500

    return jsonify({
        'message': 'File successfully uploaded and processed',
        'filename': filename,
        'script_output': result.stdout,
        'script_error': result.stderr,
        'images': existing_images
    }), 200

@app.route('/detect', methods=['POST'])
def handle_detect():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join('uploads', filename)

    # Ensure the uploads directory exists
    os.makedirs('uploads', exist_ok=True)

    file.save(file_path)

    print("Running detect_revert.py script...")
    python_path = sys.executable  # Use the current Python interpreter
    train_dir = 'Fish-DataSets/fish/train'
    val_dir = 'Fish-DataSets/fish/val'
    result = subprocess.run([python_path, 'detect_revert.py', train_dir, val_dir, file_path],
                            capture_output=True, text=True, encoding='utf-8')

    print("Script stdout:", result.stdout)
    print("Script stderr:", result.stderr)

    # Check if images were created
    image_paths = [
        'generatedimages/detected_reverted.png',
    ]

    existing_images = ['uploads/' + filename] + [path for path in image_paths if os.path.exists(path)]

    # Extract the adversarial detection result
    detection_result = "Unknown"
    for line in result.stdout.split('\n'):
        if line.startswith("Original image:"):
            detection_result = line + "\n"  # Start with the original image result
        elif line.startswith("Reverted image:"):
            detection_result += line  # Add the reverted image result
            break

    if result.returncode != 0:
        return jsonify({
            'error': 'Script failed',
            'script_output': result.stdout,
            'script_error': result.stderr
        }), 500

    return jsonify({
        'message': 'File successfully uploaded and processed',
        'filename': filename,
        'script_output': result.stdout,
        'script_error': result.stderr,
        'images': existing_images,
        'adversarial_result': detection_result
    }), 200

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/generatedimages/<filename>')
def generated_image(filename):
    return send_from_directory('generatedimages', filename)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)