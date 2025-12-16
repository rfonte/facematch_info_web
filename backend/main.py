import logging
import logging.handlers
import os
import cv2
import numpy as np
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from backend.face_utils import (
    get_face_encodings,
    save_known_faces,
    load_known_faces,
    find_matching_faces,
    analyze_face,
)

# --- Logging Setup ---
def setup_loggers():
    LOG_DIR = "logs"
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    app_logger = logging.getLogger('app')
    app_logger.setLevel(logging.INFO)
    app_handler = logging.handlers.RotatingFileHandler(
        os.path.join(LOG_DIR, "app.log"), maxBytes=2 * 1024 * 1024, backupCount=5
    )
    app_handler.setFormatter(formatter)
    app_logger.addHandler(app_handler)
    app_logger.addHandler(logging.StreamHandler())

    analysis_logger = logging.getLogger('analysis')
    analysis_logger.setLevel(logging.INFO)
    analysis_handler = logging.handlers.RotatingFileHandler(
        os.path.join(LOG_DIR, "analysis.log"), maxBytes=5 * 1024 * 1024, backupCount=5
    )
    analysis_handler.setFormatter(formatter)
    analysis_logger.addHandler(analysis_handler)

setup_loggers()
app_logger = logging.getLogger('app')
analysis_logger = logging.getLogger('analysis')

app = FastAPI()

# Carrega rostos conhecidos
try:
    known_faces_encodings, known_faces_names = load_known_faces()
    app_logger.info(f"Loaded {len(known_faces_names)} known faces.")
except Exception as e:
    app_logger.error(f"Could not load known faces on startup: {e}", exc_info=True)
    known_faces_encodings, known_faces_names = [], []

# Monta diretório do frontend
try:
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
except Exception as e:
    app_logger.error(f"Erro ao montar o frontend: {e}", exc_info=True)

@app.get("/")
async def root():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(status_code=404, content={"error": "index.html not found"})

def generate_frames():
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        app_logger.error("Não foi possível abrir a câmera.")
        return

    frame_count = 0
    while True:
        success, frame = video_capture.read()
        if not success:
            app_logger.warning("Erro ao capturar frame.")
            break

        frame_count += 1
        try:
            face_locations, face_names = find_matching_faces(
                frame, known_faces_encodings, known_faces_names
            )

            if frame_count % 30 == 0 and face_names:
                analysis_logger.info(f"Frame {frame_count}: Detected faces - {face_names}")

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

        except Exception as e:
            app_logger.error(f"Erro no processamento de frame: {e}", exc_info=True)
            break

    video_capture.release()

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/register_face")
async def register_face(name: str = Form(...), photo: UploadFile = File(...)):
    app_logger.info(f"Registrando rosto: {name}")
    try:
        image_bytes = await photo.read()
        npimg = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        face_encodings = get_face_encodings(image)
        if len(face_encodings) == 1:
            save_known_faces([(name, face_encodings[0])])
            global known_faces_encodings, known_faces_names
            known_faces_encodings, known_faces_names = load_known_faces()
            return JSONResponse({"status": "success", "message": "Rosto registrado com sucesso."})
        elif len(face_encodings) > 1:
            return JSONResponse({"status": "error", "message": "Mais de um rosto detectado."}, status_code=400)
        else:
            return JSONResponse({"status": "error", "message": "Nenhum rosto detectado."}, status_code=400)

    except Exception as e:
        app_logger.error(f"Erro ao registrar rosto: {e}", exc_info=True)
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/clear_faces")
async def clear_faces():
    try:
        if os.path.exists("known_faces.json"):
            os.remove("known_faces.json")
        global known_faces_encodings, known_faces_names
        known_faces_encodings, known_faces_names = [], []
        return JSONResponse({"status": "success", "message": "Rostos apagados com sucesso."})
    except Exception as e:
        app_logger.error(f"Erro ao limpar rostos: {e}", exc_info=True)
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    app_logger.info(f"Analisando imagem: {file.filename}")
    try:
        image_bytes = await file.read()
        result = analyze_face(image_bytes)

        if "error" not in result:
            # Salva imagem em logs/imagens/<data>/imagem.jpg
            date_folder = datetime.now().strftime("%Y-%m-%d")
            save_dir = os.path.join("logs", "imagens", date_folder)
            os.makedirs(save_dir, exist_ok=True)

            img_array = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            image_name = f"{datetime.now().strftime('%H-%M-%S')}_{file.filename}"
            image_path = os.path.join(save_dir, image_name)
            cv2.imwrite(image_path, img)

            analysis_logger.info(f"Análise concluída: {result}")
            return JSONResponse(result)

        else:
            app_logger.warning(f"Erro na análise da imagem: {result['error']}")
            return JSONResponse(result, status_code=400)

    except Exception as e:
        app_logger.error(f"Erro geral ao processar imagem: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)
