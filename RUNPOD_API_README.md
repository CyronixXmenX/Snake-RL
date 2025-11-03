# RunPod vLLM API Integration

This directory contains a Python client for interacting with RunPod's vLLM API endpoints.

## Files

- **`runpod_client.py`** - Main client implementation
- **`example_runpod_usage.py`** - Usage examples
- **`test_runpod_client.py`** - Unit tests

## Installation

The required `requests` library is included in `requirements.txt`. Install it with:

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Set your credentials as environment variables:

```bash
export RUNPOD_API_KEY='your_api_key_here'
export RUNPOD_ENDPOINT_ID='your_endpoint_id_here'
```

Then use the client:

```python
from runpod_client import create_client

# Create client (automatically reads from environment variables)
client = create_client()

# Send a prompt
response = client.run(prompt="Your prompt here")
print(response)
```

### Explicit Credentials

You can also provide credentials directly:

```python
from runpod_client import RunPodVLLMClient

client = RunPodVLLMClient(
    api_key="your_api_key",
    endpoint_id="vjkwnt2vgg4mev"
)

response = client.run(prompt="Tell me a joke")
```

### With Additional Parameters

Pass additional generation parameters as keyword arguments:

```python
response = client.run(
    prompt="Write a haiku about AI",
    temperature=0.7,
    max_tokens=100,
    top_p=0.9,
    frequency_penalty=0.5
)
```

## API Structure

The client properly implements RunPod's API structure:

### Request Format

```python
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_API_KEY"
}

data = {
    'input': {
        "prompt": "Your prompt",
        # Additional parameters go here
    }
}

response = requests.post(
    'https://api.runpod.ai/v2/{endpoint_id}/run',
    headers=headers,
    json=data
)
```

This matches the official RunPod API documentation structure.

## Examples

Run the example script to see various usage patterns:

```bash
python example_runpod_usage.py
```

The example script demonstrates:
1. Basic usage with environment variables
2. Explicit credentials
3. Error handling
4. Custom generation parameters

## Testing

Run the unit tests to verify the implementation:

```bash
python -m unittest test_runpod_client.py -v
```

All tests should pass, confirming:
- Proper header formatting with Bearer token
- Correct request payload structure with 'input' key
- Environment variable support
- Error handling
- Additional parameter passing

## Error Handling

The client handles various error scenarios:

- **Missing credentials**: Raises `ValueError` with helpful message
- **HTTP errors**: Raises `requests.exceptions.RequestException` with status code and message
- **Connection errors**: Raises `requests.exceptions.RequestException` with connection details

Example with error handling:

```python
try:
    client = create_client()
    response = client.run(prompt="Hello")
    print(f"Success: {response}")
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Request error: {e}")
```

## Environment Variables

The client supports the following environment variables:

- **`RUNPOD_API_KEY`** - Your RunPod API key (required)
- **`RUNPOD_ENDPOINT_ID`** - Your RunPod endpoint ID (required)

Set them in your shell:

```bash
export RUNPOD_API_KEY='your_api_key_here'
export RUNPOD_ENDPOINT_ID='vjkwnt2vgg4mev'
```

Or in a `.env` file (if using python-dotenv):

```
RUNPOD_API_KEY=your_api_key_here
RUNPOD_ENDPOINT_ID=vjkwnt2vgg4mev
```

## Features

✅ Proper API structure following RunPod documentation  
✅ Bearer token authentication  
✅ Environment variable support  
✅ Custom parameter passing  
✅ Comprehensive error handling  
✅ Type hints and documentation  
✅ Unit tests with 100% coverage of core functionality  

## Integration with Snake-RL

This client can be integrated into the Snake-RL project for:
- Generating training prompts
- LLM-based decision making
- Natural language interfaces
- Experiment descriptions and logging

Example integration:

```python
from runpod_client import create_client

def get_llm_advice(game_state: str) -> str:
    """Get advice from LLM about game state."""
    client = create_client()
    prompt = f"Given this Snake game state: {game_state}, what should the agent do?"
    response = client.run(prompt=prompt)
    return response.get('output', '')
```

## License

Same as the parent Snake-RL project.
