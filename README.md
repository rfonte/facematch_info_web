# FaceMatch Info Web

FaceMatch Info Web is a web application that provides face analysis and recognition functionalities.

## Features

- **Face Analysis**: Upload an image to analyze a face and determine attributes like age, gender, dominant emotion, and race.
- **Real-time Face Recognition**: Use your webcam to get a real-time video stream with recognized faces.
- **Face Registration**: Register new faces by uploading a photo and providing a name.

## Setup and Usage

### Prerequisites

- Python 3.7+
- Pip

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/facematch_info_web.git
   cd facematch_info_web
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: You might need to install `dlib` and `cmake` separately depending on your system.*

### Running the Application

1. **Start the backend server:**
   ```bash
   uvicorn backend.main:app --reload
   ```

2. **Open your browser:**
   Navigate to `http://127.0.0.1:8000` to use the application.

## API Endpoints

- `GET /`: Serves the main frontend page.
- `GET /video_feed`: Provides a real-time video stream for face recognition.
- `POST /register_face`: Registers a new face.
- `POST /clear_faces`: Deletes all known faces.
- `POST /analyze`: Analyzes an uploaded image.
