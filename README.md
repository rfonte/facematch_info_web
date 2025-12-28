# FaceMatch Info Web

FaceMatch Info Web is a web application that provides face analysis and recognition functionalities using FastAPI for the backend and a simple HTML frontend.

## Features

- **Face Analysis**: Upload an image to analyze a face and determine attributes like age, gender, dominant emotion, and race.
- **Real-time Face Recognition**: Use your webcam to get a real-time video stream with recognized faces.
- **Face Registration**: Register new faces by uploading a photo and providing a name.

## Project Structure

```
facematch_info_web/
├── src/
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── face_utils.py
│   │   └── main.py
│   └── frontend/
│       └── index.html
├── static/              # Static files (CSS, JS, images)
├── templates/           # HTML templates
├── config/              # Configuration files
├── data/                # Data files (e.g., known faces)
├── docs/                # Documentation
├── scripts/             # Utility scripts
├── tests/               # Unit and integration tests
│   ├── test_face_utils.py
│   ├── test_folder_structure.py
│   └── test_main.py
├── logs/                # Runtime logs (excluded from version control)
├── .gitignore
├── pyproject.toml       # Project configuration
├── Makefile             # Development tasks
├── README.md
├── requirements.txt
└── .env                 # Environment variables (excluded from version control)
```

## Setup and Usage

### Prerequisites

- Python 3.7+
- Pip
- Git

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
   _Note: You might need to install `dlib` and `cmake` separately depending on your system. For Windows, you can use `pip install dlib` or follow the [dlib installation guide](https://pypi.org/project/dlib/)._

### Configuration

- Copy `.env.example` to `.env` and configure environment variables if needed (e.g., for database connections or API keys).
- Adjust settings in `config/` as necessary.

### Running the Application

1. **Start the backend server:**

   ```bash
   uvicorn src.backend.main:app --reload
   ```

2. **Open your browser:**
   Navigate to `http://127.0.0.1:8000` to use the application.

### Development

- Use `make dev` to start the development server (if Makefile is configured).
- Run tests with `make test` or `pytest`.
- Lint code with `make lint` (if configured).

## API Endpoints

- `GET /`: Serves the main frontend page.
- `GET /video_feed`: Provides a real-time video stream for face recognition.
- `POST /register_face`: Registers a new face (requires `name` and `photo`).
- `POST /clear_faces`: Deletes all known faces.
- `POST /analyze`: Analyzes an uploaded image and returns face attributes.

## Testing

Run the test suite with:

```bash
pytest
```

Or use the Makefile:

```bash
make test
```

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/new-feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Create a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/) for the backend.
- Uses [DeepFace](https://github.com/serengil/deepface) for face analysis.
- OpenCV for image processing.

## Note: Google Drive integration reverted

An earlier experiment added a Google Drive upload/Picker integration. That change
was reverted. The core backend remains focused on direct file uploads and
camera input. Any temporary Drive-related utilities or tests were disabled or
removed during the revert.
