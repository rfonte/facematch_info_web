import unittest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
import os
import json

# We need to import the functions from the script we are testing
from backend import face_utils

class TestFaceUtils(unittest.TestCase):

    def setUp(self):
        """Set up test environment before each test."""
        # A dummy 1x1 pixel image for testing
        self.dummy_image_bytes = cv2.imencode('.jpg', np.zeros((1, 1, 3), dtype=np.uint8))[1].tobytes()
        self.dummy_image_np = np.zeros((10, 10, 3), dtype=np.uint8)
        self.dummy_embedding = [0.1] * 128  # DeepFace models return embeddings, often lists of floats

    @patch('cv2.imdecode')
    @patch('deepface.DeepFace.analyze')
    def test_analyze_face_success(self, mock_deepface_analyze, mock_imdecode):
        """Test analyze_face with a successful detection."""
        mock_imdecode.return_value = self.dummy_image_np
        mock_deepface_analyze.return_value = [{'age': 30, 'dominant_gender': 'Man'}]
        
        result = face_utils.analyze_face(self.dummy_image_bytes)
        
        self.assertEqual(result['age'], 30)
        self.assertEqual(result['dominant_gender'], 'Man')
        mock_deepface_analyze.assert_called_once()

    @patch('cv2.imdecode')
    def test_analyze_face_invalid_image(self, mock_imdecode):
        """Test analyze_face with invalid image data."""
        mock_imdecode.return_value = None
        result = face_utils.analyze_face(b'invalid image')
        self.assertIn('error', result)
        self.assertEqual(result['error'], 'Invalid image format')

    @patch('deepface.DeepFace.represent')
    def test_get_face_encodings(self, mock_deepface_represent):
        """Test getting face encodings from an image."""
        mock_deepface_represent.return_value = [{'embedding': self.dummy_embedding}]
        
        encodings = face_utils.get_face_encodings(self.dummy_image_np)
        
        self.assertEqual(len(encodings), 1)
        self.assertEqual(encodings[0], self.dummy_embedding)

    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_known_faces_no_file(self, mock_file, mock_exists):
        """Test loading faces when the JSON file doesn't exist."""
        mock_exists.return_value = False
        encodings, names = face_utils.load_known_faces()
        self.assertEqual(encodings, [])
        self.assertEqual(names, [])

    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_known_faces_corrupt_file(self, mock_file, mock_exists):
        """Test loading faces from a corrupt JSON file."""
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = "this is not json"
        # mock json.load to raise an error
        with patch('json.load', side_effect=json.JSONDecodeError("err", "doc", 0)):
            encodings, names = face_utils.load_known_faces()
        self.assertEqual(encodings, [])
        self.assertEqual(names, [])

    @patch('os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open)
    def test_load_known_faces_success(self, mock_file, mock_exists):
        """Test loading faces successfully from a JSON file."""
        test_data = [{'name': 'John Doe', 'encoding': self.dummy_embedding}]
        mock_file.return_value.read.return_value = json.dumps(test_data)
        with patch('json.load', return_value=test_data):
            encodings, names = face_utils.load_known_faces()
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0], 'John Doe')
        self.assertEqual(encodings[0], self.dummy_embedding)

    @patch('os.path.exists', return_value=False)
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_save_known_faces_new_file(self, mock_json_dump, mock_file, mock_exists):
        """Test saving faces to a new file."""
        new_faces = [('Jane Doe', self.dummy_embedding)]
        face_utils.save_known_faces(new_faces)
        
        mock_file.assert_called_with(face_utils.KNOWN_FACES_FILE, "w")
        mock_json_dump.assert_called_once()
        # Check that the data passed to json.dump is correct
        args, _ = mock_json_dump.call_args
        self.assertEqual(args[0][0]['name'], 'Jane Doe')
        self.assertEqual(args[0][0]['encoding'], self.dummy_embedding)

    @patch('deepface.DeepFace.extract_faces')
    def test_find_matching_faces_no_faces_in_frame(self, mock_extract_faces):
        """Test finding matches in a frame with no faces."""
        mock_extract_faces.return_value = []
        locations, names = face_utils.find_matching_faces(self.dummy_image_np, [], [])
        self.assertEqual(locations, [])
        self.assertEqual(names, [])

    @patch('deepface.commons.distance.findCosineDistance', return_value=0.1) # Known face
    @patch('deepface.DeepFace.represent')
    @patch('deepface.DeepFace.extract_faces')
    def test_find_matching_faces_one_known_face(self, mock_extract_faces, mock_represent, mock_distance):
        """Test finding a known face in the frame."""
        # Setup mock return values
        mock_extract_faces.return_value = [{'facial_area': {'x': 0, 'y': 0, 'w': 10, 'h': 10}, 'confidence': 0.99, 'face': self.dummy_image_np}]
        mock_represent.return_value = [{'embedding': self.dummy_embedding}]
        
        known_encodings = [self.dummy_embedding]
        known_names = ['John Doe']
        
        locations, names = face_utils.find_matching_faces(self.dummy_image_np, known_encodings, known_names)
        
        self.assertEqual(len(locations), 1)
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0], 'John Doe')
        mock_distance.assert_called_with(self.dummy_embedding, self.dummy_embedding)

    @patch('deepface.commons.distance.findCosineDistance', return_value=0.9) # Unknown face
    @patch('deepface.DeepFace.represent')
    @patch('deepface.DeepFace.extract_faces')
    def test_find_matching_faces_one_unknown_face(self, mock_extract_faces, mock_represent, mock_distance):
        """Test finding an unknown face in the frame."""
        mock_extract_faces.return_value = [{'facial_area': {'x': 0, 'y': 0, 'w': 10, 'h': 10}, 'confidence': 0.99, 'face': self.dummy_image_np}]
        mock_represent.return_value = [{'embedding': [0.9]*128}] # Different embedding
        
        known_encodings = [self.dummy_embedding]
        known_names = ['John Doe']
        
        locations, names = face_utils.find_matching_faces(self.dummy_image_np, known_encodings, known_names)
        
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0], 'Unknown')

if __name__ == '__main__':
    # Need to import cv2 here for the dummy image bytes to be created.
    import cv2
    unittest.main()
