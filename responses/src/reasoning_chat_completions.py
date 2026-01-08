"""
Examples used to explore Chat Completions API's reasoning object with different inference providers.
"""

import json
import os
from openai import OpenAI


MESSAGES = [
    {
        "role": "user",
        "content": "How many occurrences of letter r are in strawberry?",
    }
]


def test_reasoning_with_ollama():
    """
    Observation: an additional reasoning field is returned.
    """

    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="random",
    )

    print("Testing reasoning with ollama")

    response = client.chat.completions.create(
        model="gpt-oss:latest",
        messages=MESSAGES,
        reasoning_effort="low",
        max_completion_tokens=200,
    )
    
    print(response.model_dump_json(indent=2))


def test_reasoning_with_openai():
    """
    Observation: reasoning tokens are not returned.
    """

    client = OpenAI()

    print("Testing reasoning with openai")

    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=MESSAGES,
        reasoning_effort="low",
        max_completion_tokens=400,
    )
    
    print(response.model_dump_json(indent=2))


def test_reasoning_with_vllm():
    """
    Observation: additional reasoning and reasoning_content (deprecated) fields are returned.
    """

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="random",
    )

    print("Testing reasoning with vllm")

    response = client.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=MESSAGES,
        reasoning_effort="low",
        max_completion_tokens=200,
    )
    
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    openai_token = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_token:
        print("OPENAI_API_KEY is not set in env")
        exit(1)

    test_reasoning_with_ollama()

    test_reasoning_with_openai()

    test_reasoning_with_vllm()
