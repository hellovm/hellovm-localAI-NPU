import unittest
import json
from unittest.mock import patch, MagicMock
import sys
import os

# Ensure backend can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from backend.app import app
from backend.services.models import get_recommended_models

class TestZImageIntegration(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_recommendation_contains_z_image(self):
        """Test that Z-Image-Turbo is in the recommended T2I models"""
        models = get_recommended_models()
        t2i_ids = [m['id'] for m in models['t2i']]
        found = any("z-image" in mid.lower() for mid in t2i_ids)
        self.assertTrue(found, "Z-Image-Turbo should be in recommended T2I models")

    @patch('backend.services.inference.load_t2i_pipeline')
    @patch('backend.services.inference.t2i_generate')
    def test_z_image_default_params(self, mock_gen, mock_load):
        """Test that Z-Image model triggers correct default parameters (steps=8, guidance=0.0)"""
        import numpy as np
        mock_output = MagicMock()
        img_array = np.zeros((512, 512, 3), dtype=np.uint8)
        mock_output.data = [img_array]
        mock_gen.return_value = mock_output
        
        mock_load.return_value = MagicMock()
        
        payload = {
            "model_path": "Tongyi-MAI/Z-Image-Turbo",
            "prompt": "test prompt",
            "width": 512,
            "height": 512
        }
        
        with patch('backend.app.MODELS_DIR') as MockModelsDir:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            MockModelsDir.__truediv__.return_value = mock_path
            
            mock_index = MagicMock()
            mock_index.exists.return_value = True
            mock_path.__truediv__.return_value = mock_index
            
            response = self.app.post('/api/image/generate', 
                                   data=json.dumps(payload),
                                   content_type='application/json')
            
            self.assertEqual(response.status_code, 200)
            args, kwargs = mock_gen.call_args
            self.assertEqual(kwargs.get('steps'), 8, "Should use 8 steps for Z-Image")
            self.assertEqual(kwargs.get('guidance_scale'), 0.0, "Should use 0.0 guidance for Z-Image")

    @patch('backend.services.inference.load_t2i_pipeline')
    @patch('backend.services.inference.t2i_generate')
    def test_standard_model_defaults(self, mock_gen, mock_load):
        """Test that standard models use standard defaults"""
        import numpy as np
        mock_output = MagicMock()
        img_array = np.zeros((512, 512, 3), dtype=np.uint8)
        mock_output.data = [img_array]
        mock_gen.return_value = mock_output
        
        mock_load.return_value = MagicMock()
        
        payload = {
            "model_path": "runwayml/stable-diffusion-v1-5",
            "prompt": "test prompt",
            "width": 512,
            "height": 512
        }
        
        with patch('backend.app.MODELS_DIR') as MockModelsDir:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            MockModelsDir.__truediv__.return_value = mock_path
            
            mock_index = MagicMock()
            mock_index.exists.return_value = True
            mock_path.__truediv__.return_value = mock_index
            
            response = self.app.post('/api/image/generate', 
                                   data=json.dumps(payload),
                                   content_type='application/json')
            
            self.assertEqual(response.status_code, 200)
            args, kwargs = mock_gen.call_args
            self.assertEqual(kwargs.get('steps'), 30, "Should use 30 steps for standard model")
            self.assertEqual(kwargs.get('guidance_scale'), 7.5, "Should use 7.5 guidance for standard model")

if __name__ == '__main__':
    unittest.main()
