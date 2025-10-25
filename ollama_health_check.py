import requests
import logging

OLLAMA_API_URL = "http://localhost:11434/v1/chat/completions"
OLLAMA_MODEL_NAME = "mistral"  # Replace with your default model or import from config if needed

def check_ollama_model():
    """
    Checks if Ollama is running and the specified model is available.
    Returns True if healthy, False otherwise.
    """
    try:
        # Use a minimal prompt to test
        payload = {
            "model": OLLAMA_MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a test system."},
                {"role": "user", "content": "Hello"}
            ],
            "stream": False
        }
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=10)
        if response.status_code == 200:
            logging.info(f"Ollama health check passed for model '{OLLAMA_MODEL_NAME}'.")
            return True
        else:
            logging.error(f"Ollama health check failed: {response.status_code} {response.text}")
            return False
    except Exception as e:
        logging.error(f"Ollama health check failed: {e}")
        return False

if __name__ == "__main__":
    healthy = check_ollama_model()
    print(f"Ollama health check: {'OK' if healthy else 'FAILED'}")
