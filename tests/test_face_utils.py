import os
import sys
import numpy as np
import cv2
import pytest

# Ajusta o sys.path para importar face_utils.py corretamente
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from face_utils import (
    analyze_face,
    get_face_encodings,
    load_known_faces,
    save_known_faces,
    find_matching_faces
)

def test_analyze_face_with_valid_image():
    image_path = os.path.join(os.path.dirname(__file__), 'sample_image.jpg')
    if not os.path.exists(image_path):
        pytest.skip("Imagem de teste 'sample_image.jpg' ausente.")

    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    result = analyze_face(image_bytes)
    assert 'age' in result
    assert 'dominant_gender' in result
    assert 'dominant_emotion' in result
    assert 'dominant_race' in result

def test_analyze_face_with_invalid_image():
    invalid_bytes = b'this_is_not_an_image'
    result = analyze_face(invalid_bytes)
    assert 'error' in result

def test_get_face_encodings():
    image_path = os.path.join(os.path.dirname(__file__), 'sample_image.jpg')
    if not os.path.exists(image_path):
        pytest.skip("Imagem de teste 'sample_image.jpg' ausente.")

    image = cv2.imread(image_path)
    encodings = get_face_encodings(image)
    assert isinstance(encodings, list)

def test_save_and_load_known_faces(tmp_path):
    test_file = tmp_path / "test_faces.json"
    test_data = [("Rodrigo", [0.1] * 2622)]  # VGG-Face embedding size

    # Redefine a constante global temporariamente
    import face_utils
    face_utils.KNOWN_FACES_FILE = str(test_file)

    save_known_faces(test_data)
    encodings, names = load_known_faces()

    assert len(encodings) == 1
    assert len(names) == 1
    assert names[0] == "Rodrigo"
    assert isinstance(encodings[0], list)

def test_find_matching_faces():
    image_path = os.path.join(os.path.dirname(__file__), 'sample_image.jpg')
    if not os.path.exists(image_path):
        pytest.skip("Imagem de teste 'sample_image.jpg' ausente.")

    frame = cv2.imread(image_path)
    face_locations, face_names = find_matching_faces(frame, [], [])

    assert isinstance(face_locations, list)
    assert isinstance(face_names, list)
