"""
Example usage of RunPod vLLM API client.

This script demonstrates how to use the RunPodVLLMClient to interact
with RunPod's vLLM API endpoints.
"""

import os
from runpod_client import RunPodVLLMClient, create_client


def example_basic_usage():
    """Basic example using environment variables."""
    print("=== Example 1: Basic Usage ===")
    
    # Create client (reads from RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID env vars)
    try:
        client = create_client()
        
        # Send a simple prompt
        response = client.run(prompt="What is the capital of France?")
        print(f"Response: {response}")
    except ValueError as e:
        print(f"Error: {e}")
        print("\nPlease set environment variables:")
        print("  export RUNPOD_API_KEY='your_api_key_here'")
        print("  export RUNPOD_ENDPOINT_ID='your_endpoint_id_here'")


def example_explicit_credentials():
    """Example with explicit credentials."""
    print("\n=== Example 2: Explicit Credentials ===")
    
    # For demonstration - replace with your actual credentials
    api_key = os.environ.get("RUNPOD_API_KEY", "YOUR_API_KEY")
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID", "vjkwnt2vgg4mev")
    
    try:
        client = RunPodVLLMClient(
            api_key=api_key,
            endpoint_id=endpoint_id
        )
        
        # Send a prompt with additional parameters
        response = client.run(
            prompt="Tell me a short joke about programming",
            temperature=0.7,
            max_tokens=100
        )
        print(f"Response: {response}")
    except ValueError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"Request error: {e}")


def example_with_error_handling():
    """Example with comprehensive error handling."""
    print("\n=== Example 3: With Error Handling ===")
    
    api_key = os.environ.get("RUNPOD_API_KEY")
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")
    
    if not api_key or not endpoint_id:
        print("Please set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID environment variables")
        return
    
    try:
        client = RunPodVLLMClient(api_key=api_key, endpoint_id=endpoint_id)
        
        prompts = [
            "What is machine learning?",
            "Explain reinforcement learning in simple terms",
            "What is a neural network?"
        ]
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\nPrompt {i}: {prompt}")
            try:
                response = client.run(prompt=prompt, timeout=30)
                print(f"Success: {response.get('status', 'unknown')}")
                # Print relevant parts of the response
                if 'output' in response:
                    print(f"Output: {response['output']}")
            except Exception as e:
                print(f"Error processing prompt: {e}")
                
    except ValueError as e:
        print(f"Client initialization error: {e}")


def example_custom_parameters():
    """Example with custom generation parameters."""
    print("\n=== Example 4: Custom Parameters ===")
    
    api_key = os.environ.get("RUNPOD_API_KEY")
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")
    
    if not api_key or not endpoint_id:
        print("Please set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID environment variables")
        return
    
    try:
        client = RunPodVLLMClient(api_key=api_key, endpoint_id=endpoint_id)
        
        # Example with various parameters
        response = client.run(
            prompt="Write a haiku about artificial intelligence",
            temperature=0.8,
            max_tokens=50,
            top_p=0.9,
            frequency_penalty=0.5
        )
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run all examples."""
    print("RunPod vLLM API Client Examples")
    print("=" * 50)
    
    example_basic_usage()
    example_explicit_credentials()
    example_with_error_handling()
    example_custom_parameters()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nNote: To use these examples, set your credentials:")
    print("  export RUNPOD_API_KEY='your_api_key_here'")
    print("  export RUNPOD_ENDPOINT_ID='your_endpoint_id_here'")


if __name__ == "__main__":
    main()
