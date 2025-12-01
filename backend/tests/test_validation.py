import unittest
import json
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Ensure backend is in path
sys.path.append(str(Path(__file__).resolve().parents[2]))

class TestAPIValidation(unittest.TestCase):
    def setUp(self):
        # Import app inside setup to avoid issues if paths aren't set yet
        from backend.app import app
        self.client = app.test_client()

    def test_chat_validation(self):
        # Test invalid temperature
        data = {
            "model_id": "test/model",
            "prompt": "hi",
            "config": {"temperature": 2.5}
        }
        resp = self.client.post("/api/infer/chat", json=data)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("temperature", resp.get_json()["message"])

        # Test invalid max_new_tokens
        data["config"] = {"max_new_tokens": 9000}
        resp = self.client.post("/api/infer/chat", json=data)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("max_new_tokens", resp.get_json()["message"])

    def test_image_validation(self):
        # Test too small dimension (must be divisible by 8 to hit this check first, or hit divisible check)
        # Let's ensure it passes divisible check but fails min dimension
        data = {
            "model_path": "test/model",
            "prompt": "a cat",
            "width": 240, # Divisible by 8 (30*8), but < 256
            "height": 512
        }
        resp = self.client.post("/api/image/generate", json=data)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Min dimension", resp.get_json()["message"])

        # Test not divisible by 8
        data["width"] = 257
        resp = self.client.post("/api/image/generate", json=data)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("divisible by 8", resp.get_json()["message"])

        # Test steps too large
        data["width"] = 512
        data["steps"] = 150
        resp = self.client.post("/api/image/generate", json=data)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Steps", resp.get_json()["message"])

    def test_download_precision_validation(self):
        data = {
            "model_id": "test/model",
            "precision": "fp32" # Invalid
        }
        resp = self.client.post("/api/image/ms_download_and_convert", json=data)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("fp16 or int8", resp.get_json()["message"])

        resp = self.client.post("/api/image/download_model", json={"hf_id": "test", "precision": "fp32"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("fp16 or int8", resp.get_json()["message"])

    def test_video_validation(self):
        data = {
            "model_path": "test/model",
            "prompt": "video",
            "seconds": 100
        }
        resp = self.client.post("/api/video/generate", json=data)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Seconds", resp.get_json()["message"])

if __name__ == "__main__":
    unittest.main()
