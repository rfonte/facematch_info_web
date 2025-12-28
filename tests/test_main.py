import os
import sys
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.main import app

from fastapi.testclient import TestClient

client = TestClient(app)

def test_root_route():
    response = client.get("/")
    assert response.status_code in [200, 404]  # depende se index.html está lá

def test_video_feed_route():
    response = client.get("/video_feed")
    assert response.status_code == 200

def test_clear_faces():
    response = client.post("/clear_faces")
    assert response.status_code == 200
    assert "success" in response.json()["status"]

def test_register_face_missing_file():
    response = client.post("/register_face", data={"name": "Test"})
    assert response.status_code == 422  # falta o arquivo

def test_analyze_image_no_file():
    response = client.post("/analyze")
    assert response.status_code == 422  # ausência do arquivo
