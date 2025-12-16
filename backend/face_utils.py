from deepface import DeepFace
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
import os
import json
from numpy import dot
from numpy.linalg import norm

KNOWN_FACES_FILE = "known_faces.json"
COSINE_THRESHOLD = 0.40  # Para VGG-Face + Cosine

# Função para calcular a distância cosseno (substitui deepface.commons.distance)
def find_cosine_distance(vec1, vec2):
    return 1 - (dot(vec1, vec2) / (norm(vec1) * norm(vec2)))

def analyze_face(image_bytes: bytes) -> Dict[str, Any]:
    """
    Analisa uma imagem facial usando DeepFace.
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image format"}

        result = DeepFace.analyze(img, actions=['age', 'gender', 'emotion', 'race'], enforce_detection=False)

        if isinstance(result, list) and len(result) > 0:
            return result[0]
        else:
            return {"error": "No face detected or analysis failed"}

    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

def get_face_encodings(image: np.ndarray) -> List[any]:
    """
    Extrai embeddings faciais de uma imagem usando DeepFace.
    """
    try:
        results = DeepFace.represent(img_path=image, enforce_detection=True, model_name='VGG-Face')
        return [result['embedding'] for result in results]
    except Exception:
        return []

def load_known_faces() -> Tuple[List[any], List[str]]:
    """
    Carrega rostos conhecidos de arquivo JSON.
    """
    if not os.path.exists(KNOWN_FACES_FILE):
        return [], []
    
    try:
        with open(KNOWN_FACES_FILE, "r") as f:
            data = json.load(f)
        
        names = [item['name'] for item in data]
        encodings = [item['encoding'] for item in data]
        return encodings, names
    except (json.JSONDecodeError, IOError):
        return [], []

def save_known_faces(new_faces: List[Tuple[str, any]]):
    """
    Salva novos rostos conhecidos no arquivo JSON.
    """
    existing_data = []
    if os.path.exists(KNOWN_FACES_FILE):
        with open(KNOWN_FACES_FILE, "r") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                pass

    for name, encoding in new_faces:
        existing_data.append({"name": name, "encoding": encoding})

    with open(KNOWN_FACES_FILE, "w") as f:
        json.dump(existing_data, f, indent=4)

def find_matching_faces(frame: np.ndarray, known_faces_encodings: List[any], known_faces_names: List[str]) -> Tuple[List[Tuple[int, int, int, int]], List[str]]:
    """
    Detecta rostos no frame e tenta encontrar correspondências com rostos conhecidos.
    """
    face_locations = []
    face_names = []
    
    try:
        extracted_faces = DeepFace.extract_faces(frame, enforce_detection=False)
    except Exception:
        return [], []

    for face_info in extracted_faces:
        if face_info['confidence'] == 0:
            continue

        x, y, w, h = face_info['facial_area'].values()
        face_locations.append((y, x + w, y + h, x))
        
        try:
            embedding_list = DeepFace.represent(
                face_info['face'],
                enforce_detection=False,
                detector_backend='skip',
                model_name='VGG-Face'
            )
            if not embedding_list:
                face_names.append("Unknown")
                continue
            
            embedding = embedding_list[0]['embedding']
        except Exception:
            face_names.append("Unknown")
            continue

        name = "Unknown"
        min_dist = float('inf')

        for i, known_encoding in enumerate(known_faces_encodings):
            distance = find_cosine_distance(embedding, known_encoding)
            
            if distance < min_dist and distance < COSINE_THRESHOLD:
                min_dist = distance
                name = known_faces_names[i]
        
        face_names.append(name)

    return face_locations, face_names
