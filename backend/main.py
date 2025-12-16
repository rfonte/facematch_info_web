import logging
import logging.handlers
import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from .face_utils import (
    get_face_encodings,
    save_known_faces,
    load_known_faces,
    find_matching_faces,
    analyze_face,
)

# --- Logging Setup ---
def setup_loggers():
    """Configures two separate loggers for the application."""
    LOG_DIR = "logs"
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # 1. Application Logger
    app_logger = logging.getLogger('app')
    app_logger.setLevel(logging.INFO)
    app_handler = logging.handlers.RotatingFileHandler(
        os.path.join(LOG_DIR, "app.log"), maxBytes=2*1024*1024, backupCount=5
    )
    app_handler.setFormatter(formatter)
    app_logger.addHandler(app_handler)
    app_logger.addHandler(logging.StreamHandler()) # Also log to console

    # 2. Analysis Logger
    analysis_logger = logging.getLogger('analysis')
    analysis_logger.setLevel(logging.INFO)
    analysis_handler = logging.handlers.RotatingFileHandler(
        os.path.join(LOG_DIR, "analysis.log"), maxBytes=5*1024*1024, backupCount=5
    )
    analysis_handler.setFormatter(formatter)
    analysis_logger.addHandler(analysis_handler)

setup_loggers()

# Get loggers
app_logger = logging.getLogger('app')
analysis_logger = logging.getLogger('analysis')

app_logger.info("Application starting up.")

# --- FastAPI App Initialization ---
app = FastAPI()

# Load known faces on startup
try:
    known_faces_encodings, known_faces_names = load_known_faces()
    app_logger.info(f"Loaded {len(known_faces_names)} known faces.")
except Exception as e:
    app_logger.error(f"Could not load known faces on startup: {e}", exc_info=True)
    known_faces_encodings, known_faces_names = [], []

# Mount frontend directory
try:
    frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
    frontend_dir = os.path.abspath(frontend_dir)
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
    app_logger.info(f"Mounted frontend directory at: {frontend_dir}")
except Exception as e:
    app_logger.error(f"Error mounting frontend directory: {e}", exc_info=True)

@app.get("/")
async def root():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    message = "index.html not found"
    app_logger.error(message)
    return JSONResponse(status_code=404, content={"error": message})

def generate_frames():
    """Generator for camera frames with face detection."""
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        app_logger.error("Cannot open camera.")
        return

    frame_count = 0
    while True:
        success, frame = video_capture.read()
        if not success:
            app_logger.warning("Failed to read frame from camera.")
            break

        frame_count += 1
        try:
            face_locations, face_names = find_matching_faces(
                frame, known_faces_encodings, known_faces_names
            )

            # Log analysis only periodically to avoid spamming the log
            if frame_count % 30 == 0 and face_names: # Log every 30 frames if a face is found
                 analysis_logger.info(f"Frame {frame_count}: Detected faces - {face_names}")


            for (top, right, bottom, left), name in zip(face_locations, face_names):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                app_logger.warning("Failed to encode frame.")
                continue
            
            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

        except Exception as e:
            app_logger.error(f"Error in frame generation: {e}", exc_info=True)
            break
    
    video_capture.release()
    app_logger.info("Camera released.")


@app.get("/video_feed")
def video_feed():
    """Route for video streaming."""
    app_logger.info("Video feed requested.")
    return StreamingResponse(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.post("/register_face")
async def register_face(name: str = Form(...), photo: UploadFile = File(...)):
    """Route to register a new face from an image."""
    app_logger.info(f"Received request to register face for: {name}")
    try:
        image_bytes = await photo.read()
        npimg = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        face_encodings = get_face_encodings(image)

        if len(face_encodings) == 1:
            save_known_faces([(name, face_encodings[0])])
            
            global known_faces_encodings, known_faces_names
            known_faces_encodings, known_faces_names = load_known_faces()
            
            analysis_logger.info(f"Successfully registered face for '{name}'.")
            return JSONResponse({"status": "success", "message": "Rosto registrado com sucesso."})
        
        elif len(face_encodings) > 1:
            app_logger.warning(f"Registration failed for {name}: more than one face detected.")
            return JSONResponse({
                "status": "error",
                "message": "Mais de um rosto detectado. Envie uma imagem com apenas um rosto.",
            }, status_code=400)
            
        else:
            app_logger.warning(f"Registration failed for {name}: no face detected.")
            return JSONResponse({
                "status": "error", 
                "message": "Nenhum rosto detectado na imagem."
            }, status_code=400)

    except Exception as e:
        app_logger.error(f"Error during face registration for {name}: {e}", exc_info=True)
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/clear_faces")
async def clear_faces():
    """Route to clear all known faces."""
    app_logger.info("Received request to clear all faces.")
    try:
        if os.path.exists("known_faces.json"):
            os.remove("known_faces.json")
        
        global known_faces_encodings, known_faces_names
        known_faces_encodings, known_faces_names = [], []
        
        analysis_logger.info("All registered faces have been cleared.")
        return JSONResponse({"status": "success", "message": "Todos os rostos foram removidos."})
    except Exception as e:
        app_logger.error(f"Error clearing faces: {e}", exc_info=True)
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze an uploaded image for facial features."""
    app_logger.info(f"Received request to analyze image: {file.filename}")
    try:
        image_bytes = await file.read()
        result = analyze_face(image_bytes)
        
        if "error" not in result:
            analysis_logger.info(f"Analysis successful for {file.filename}. Result: {result}")
            return JSONResponse(result)
        else:
            app_logger.error(f"Analysis failed for {file.filename}: {result['error']}")
            return JSONResponse(result, status_code=400)

    except Exception as e:
        app_logger.error(f"Failed to process image {file.filename}: {e}", exc_info=True)
        return JSONResponse({"error": f"Failed to process image: {str(e)}"}, status_code=500)
