THIS IS A DELIBERATE SYNTAX ERROR TO TEST THE FILE PATH

from deepface import DeepFace
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
import os
import json
from deepface.commons.distance import findCosineDistance

KNOWN_FACES_FILE = "known_faces.json"
# Threshold for VGG-Face model with Cosine distance. Other models/distances will need different thresholds.
# You can find thresholds in the DeepFace source for deepface.commons.distance
COSINE_THRESHOLD = 0.40 


def analyze_face(image_bytes: bytes) -> Dict[str, Any]:
    """
    Analyze a face from image bytes using DeepFace.

    Args:
        image_bytes: Raw image data as bytes

    Returns:
        Dict containing analysis results
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
    Get face encodings (representations) from an image using DeepFace.
    """
    try:
        # DeepFace.represent returns a list of dictionaries, one for each detected face.
        results = DeepFace.represent(img_path=image, enforce_detection=True, model_name='VGG-Face')
        # We are interested in the 'embedding' which is the face encoding.
        return [result['embedding'] for result in results]
    except Exception:
        # Return an empty list if no faces are detected or an error occurs.
        return []

def load_known_faces() -> Tuple[List[any], List[str]]:
    """Loads known face encodings and names from a JSON file."""
    if not os.path.exists(KNOWN_FACES_FILE):
        return [], []
    
    try:
        with open(KNOWN_FACES_FILE, "r") as f:
            data = json.load(f)
        
        names = [item['name'] for item in data]
        encodings = [item['encoding'] for item in data]
        return encodings, names
    except (json.JSONDecodeError, IOError):
        # If the file is empty, corrupt, or can't be read, return empty lists.
        return [], []

def save_known_faces(new_faces: List[Tuple[str, any]]):
    """Saves new face encodings and names to the JSON file."""
    existing_data = []
    if os.path.exists(KNOWN_FACES_FILE):
        with open(KNOWN_FACES_FILE, "r") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                # File is corrupt or empty, will be overwritten.
                pass

    for name, encoding in new_faces:
        existing_data.append({"name": name, "encoding": encoding})

    with open(KNOWN_FACES_FILE, "w") as f:
        json.dump(existing_data, f, indent=4)


def find_matching_faces(frame: np.ndarray, known_faces_encodings: List[any], known_faces_names: List[str]) -> Tuple[List[Tuple[int, int, int, int]], List[str]]:
    """Finds matching faces in a frame by comparing them to known faces."""
    face_locations = []
    face_names = []
    
    try:
        # Extract faces from the current frame.
        # This returns a list of dicts, each with the face image and its location.
        extracted_faces = DeepFace.extract_faces(frame, enforce_detection=False)
    except Exception:
        # If face extraction fails, return empty lists.
        return [], []

    # Iterate over each face found in the frame.
    for face_info in extracted_faces:
        if face_info['confidence'] == 0:
            continue

        # Get the location of the face.
        x, y, w, h = face_info['facial_area'].values()
        face_locations.append((y, x + w, y + h, x))
        
        # Get the embedding for the current face.
        try:
            # We pass the cropped face image directly to represent.
            # detector_backend='skip' is important because we've already detected the face.
            embedding_list = DeepFace.represent(face_info['face'], enforce_detection=False, detector_backend='skip', model_name='VGG-Face')
            if not embedding_list:
                face_names.append("Unknown")
                continue
            
            embedding = embedding_list[0]['embedding']
        except Exception:
            face_names.append("Unknown")
            continue

        name = "Unknown"
        min_dist = float('inf')

        # Compare the current face's embedding with all known face embeddings.
        for i, known_encoding in enumerate(known_faces_encodings):
            distance = findCosineDistance(embedding, known_encoding)
            
            if distance < min_dist and distance < COSINE_THRESHOLD:
                min_dist = distance
                name = known_faces_names[i]
        
        face_names.append(name)

    return face_locations, face_names
