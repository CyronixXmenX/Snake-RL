"""
RunPod vLLM API Client.

This module provides a client for interacting with RunPod's vLLM API endpoints.
It properly implements the API request structure as specified in RunPod's documentation.
"""

import os
from typing import Optional, Dict, Any

import requests


class RunPodVLLMClient:
    """
    Client for RunPod vLLM API.
    
    This client handles authentication and request formatting for RunPod's
    vLLM API endpoints following their official API structure.
    
    Attributes:
        api_key: RunPod API key for authentication
        endpoint_id: RunPod endpoint ID (e.g., 'vjkwnt2vgg4mev')
        base_url: Base URL for RunPod API
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        base_url: str = "https://api.runpod.ai/v2"
    ):
        """
        Initialize RunPod vLLM client.
        
        Args:
            api_key: RunPod API key. If not provided, reads from RUNPOD_API_KEY env var.
            endpoint_id: RunPod endpoint ID. If not provided, reads from RUNPOD_ENDPOINT_ID env var.
            base_url: Base URL for RunPod API (default: https://api.runpod.ai/v2)
        
        Raises:
            ValueError: If api_key or endpoint_id is not provided and not found in environment
        """
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        self.endpoint_id = endpoint_id or os.environ.get("RUNPOD_ENDPOINT_ID")
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError(
                "API key is required. Provide it via 'api_key' parameter or "
                "set RUNPOD_API_KEY environment variable."
            )
        
        if not self.endpoint_id:
            raise ValueError(
                "Endpoint ID is required. Provide it via 'endpoint_id' parameter or "
                "set RUNPOD_ENDPOINT_ID environment variable."
            )
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get HTTP headers for API requests.
        
        Returns:
            Dictionary containing Content-Type and Authorization headers
        """
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def run(
        self,
        prompt: str,
        timeout: Optional[int] = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a prompt to the RunPod vLLM API endpoint.
        
        This method follows RunPod's API structure where the request payload
        contains an 'input' key with the prompt and any additional parameters.
        
        Args:
            prompt: The text prompt to send to the model
            timeout: Request timeout in seconds (default: 30)
            **kwargs: Additional parameters to include in the input payload
        
        Returns:
            API response as a dictionary
        
        Raises:
            requests.exceptions.RequestException: If the API request fails
            ValueError: If the response is invalid
        
        Example:
            >>> client = RunPodVLLMClient(api_key="your_key", endpoint_id="vjkwnt2vgg4mev")
            >>> response = client.run(prompt="Hello, world!")
            >>> print(response)
        """
        url = f"{self.base_url}/{self.endpoint_id}/run"
        
        # Structure the payload according to RunPod API specification
        data = {
            'input': {
                "prompt": prompt,
                **kwargs  # Allow additional parameters like temperature, max_tokens, etc.
            }
        }
        
        headers = self._get_headers()
        
        try:
            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            raise requests.exceptions.RequestException(
                f"RunPod API request failed with status {response.status_code}: {response.text}"
            ) from e
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(
                f"Failed to connect to RunPod API: {str(e)}"
            ) from e
    
    def run_sync(
        self,
        prompt: str,
        timeout: Optional[int] = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for run() method.
        
        This is an alias for run() that explicitly indicates synchronous behavior.
        For async operations, use the run() method with appropriate timeout.
        
        Args:
            prompt: The text prompt to send to the model
            timeout: Request timeout in seconds (default: 30)
            **kwargs: Additional parameters to include in the input payload
        
        Returns:
            API response as a dictionary
        """
        return self.run(prompt=prompt, timeout=timeout, **kwargs)


def create_client(
    api_key: Optional[str] = None,
    endpoint_id: Optional[str] = None
) -> RunPodVLLMClient:
    """
    Convenience function to create a RunPod vLLM client.
    
    Args:
        api_key: RunPod API key. If not provided, reads from RUNPOD_API_KEY env var.
        endpoint_id: RunPod endpoint ID. If not provided, reads from RUNPOD_ENDPOINT_ID env var.
    
    Returns:
        Initialized RunPodVLLMClient instance
    
    Example:
        >>> client = create_client()
        >>> response = client.run(prompt="Tell me a joke")
    """
    return RunPodVLLMClient(api_key=api_key, endpoint_id=endpoint_id)
