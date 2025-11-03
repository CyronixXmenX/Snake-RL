"""
Unit tests for RunPod vLLM API client.

Tests the RunPodVLLMClient implementation to ensure proper API structure
and error handling.
"""

import os
import unittest
from unittest.mock import patch, Mock
from runpod_client import RunPodVLLMClient, create_client


class TestRunPodVLLMClient(unittest.TestCase):
    """Test cases for RunPodVLLMClient."""
    
    def test_client_initialization_with_params(self):
        """Test client can be initialized with explicit parameters."""
        client = RunPodVLLMClient(
            api_key="test_key",
            endpoint_id="test_endpoint"
        )
        self.assertEqual(client.api_key, "test_key")
        self.assertEqual(client.endpoint_id, "test_endpoint")
        self.assertEqual(client.base_url, "https://api.runpod.ai/v2")
    
    def test_client_initialization_with_env_vars(self):
        """Test client can be initialized from environment variables."""
        with patch.dict(os.environ, {
            'RUNPOD_API_KEY': 'env_key',
            'RUNPOD_ENDPOINT_ID': 'env_endpoint'
        }):
            client = RunPodVLLMClient()
            self.assertEqual(client.api_key, "env_key")
            self.assertEqual(client.endpoint_id, "env_endpoint")
    
    def test_client_missing_api_key(self):
        """Test that ValueError is raised when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as context:
                RunPodVLLMClient(endpoint_id="test_endpoint")
            self.assertIn("API key is required", str(context.exception))
    
    def test_client_missing_endpoint_id(self):
        """Test that ValueError is raised when endpoint ID is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as context:
                RunPodVLLMClient(api_key="test_key")
            self.assertIn("Endpoint ID is required", str(context.exception))
    
    def test_get_headers(self):
        """Test that headers are properly formatted."""
        client = RunPodVLLMClient(
            api_key="test_key_123",
            endpoint_id="test_endpoint"
        )
        headers = client._get_headers()
        
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["Authorization"], "Bearer test_key_123")
    
    @patch('runpod_client.requests.post')
    def test_run_method_structure(self, mock_post):
        """Test that run() method creates correct API request structure."""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success", "output": "test"}
        mock_post.return_value = mock_response
        
        client = RunPodVLLMClient(
            api_key="test_key",
            endpoint_id="vjkwnt2vgg4mev"
        )
        
        # Make request
        result = client.run(prompt="Test prompt")
        
        # Verify the call was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        
        # Check URL
        self.assertEqual(
            call_args[0][0],
            "https://api.runpod.ai/v2/vjkwnt2vgg4mev/run"
        )
        
        # Check headers
        headers = call_args[1]['headers']
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["Authorization"], "Bearer test_key")
        
        # Check payload structure matches RunPod API specification
        payload = call_args[1]['json']
        self.assertIn('input', payload)
        self.assertIn('prompt', payload['input'])
        self.assertEqual(payload['input']['prompt'], "Test prompt")
        
        # Check result
        self.assertEqual(result, {"status": "success", "output": "test"})
    
    @patch('runpod_client.requests.post')
    def test_run_with_additional_params(self, mock_post):
        """Test that additional parameters are included in the request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_post.return_value = mock_response
        
        client = RunPodVLLMClient(
            api_key="test_key",
            endpoint_id="test_endpoint"
        )
        
        client.run(
            prompt="Test",
            temperature=0.7,
            max_tokens=100,
            top_p=0.9
        )
        
        # Check that additional parameters are in the input
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        
        self.assertEqual(payload['input']['prompt'], "Test")
        self.assertEqual(payload['input']['temperature'], 0.7)
        self.assertEqual(payload['input']['max_tokens'], 100)
        self.assertEqual(payload['input']['top_p'], 0.9)
    
    @patch('runpod_client.requests.post')
    def test_run_http_error(self, mock_post):
        """Test error handling for HTTP errors."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_response.raise_for_status.side_effect = Exception("HTTP Error")
        mock_post.return_value = mock_response
        
        client = RunPodVLLMClient(
            api_key="invalid_key",
            endpoint_id="test_endpoint"
        )
        
        with self.assertRaises(Exception):
            client.run(prompt="Test")
    
    def test_create_client_convenience_function(self):
        """Test the convenience function for creating a client."""
        with patch.dict(os.environ, {
            'RUNPOD_API_KEY': 'env_key',
            'RUNPOD_ENDPOINT_ID': 'env_endpoint'
        }):
            client = create_client()
            self.assertIsInstance(client, RunPodVLLMClient)
            self.assertEqual(client.api_key, "env_key")
            self.assertEqual(client.endpoint_id, "env_endpoint")
    
    @patch('runpod_client.requests.post')
    def test_run_sync_alias(self, mock_post):
        """Test that run_sync() works as an alias for run()."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_post.return_value = mock_response
        
        client = RunPodVLLMClient(
            api_key="test_key",
            endpoint_id="test_endpoint"
        )
        
        result = client.run_sync(prompt="Test prompt")
        
        # Verify call was made
        mock_post.assert_called_once()
        self.assertEqual(result, {"status": "success"})


if __name__ == '__main__':
    unittest.main()
