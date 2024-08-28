# AdversaGuard: Image Attack Analysis and Defense

## Project Description

AdversaGuard is a tool designed to generate, detect, and defend against adversarial attacks on image classification systems. This project showcases the implementation of Fast Gradient Sign Method (FGSM) for generating adversarial examples, along with detection and reversion techniques to counteract such attacks.

## Features

- **Adversarial Image Generation**: Utilizes FGSM to create adversarial examples that can fool image classification models.
- **Attack Detection**: Implements algorithms to identify potential adversarial manipulations in images.
- **Image Reversion**: Attempts to revert detected adversarial images back to their original state.
- **Interactive Web Interface**: Provides a user-friendly platform for uploading images, generating adversarial examples, and visualizing results.

## Technologies Used

- Python
- TensorFlow
- Flask
- HTML/CSS/JavaScript
- Bootstrap

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/ImageAttackAnalysis.git
   ```

2. Navigate to the project directory:
   ```
   cd ImageAttackAnalysis/FreedomGen
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Running the Application

1. Start the Flask server:
   ```
   python server.py
   ```

2. Open a web browser and navigate to `http://localhost:5000`

## Usage

1. **Generate Adversarial Images**:
   - Click on the "Generate" tab
   - Upload an image
   - View the generated adversarial examples and training history

2. **Detect and Revert Adversarial Images**:
   - Click on the "Detect" tab
   - Upload a potentially adversarial image
   - View the detection results and reverted image

## Contributing

Contributions to AdversaGuard are welcome! Please feel free to submit a Pull Request.
