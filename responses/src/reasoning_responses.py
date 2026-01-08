"""
Examples used to explore OpenAI Responses API's reasoning object with different inference providers.
"""

import json
import os
from openai import OpenAI


INPUT = "How many occurrences of letter r are in strawberry?"


def test_reasoning_with_lls():
    """
    Observation: Llama Stack returns reasoning object in the streaming events.
    """

    client = OpenAI(
        base_url="http://localhost:8321/v1/",
        api_key="random",
    )

    print("Testing reasoning with LLS")

    response = client.responses.create(
        model="ollama/gpt-oss:latest",
        input=INPUT,
        reasoning={"effort": "low"},
        max_output_tokens=200,
        stream=True,
    )
    
    for chunk in response:
        print(chunk.model_dump_json(indent=2))


def test_reasoning_with_ollama():
    """
    Observation: reasoning content is returned in the summary and encrypted content of the
    reasoning output object.
    """

    client = OpenAI(
        base_url="http://localhost:11434/v1/",
        api_key="random",
    )

    print("Testing reasoning with ollama")

    response = client.responses.create(
        model="gpt-oss:latest",
        input=INPUT,
        reasoning={"effort": "low"},
        max_output_tokens=200,
    )
    
    print(response.model_dump_json(indent=2))


def test_reasoning_with_openai():
    """
    Observation: reasoning tokens are not returned in the output object.
    """

    client = OpenAI()

    print("Testing reasoning with openai")

    response = client.responses.create(
        model="gpt-5-nano",
        input=INPUT,
        reasoning={"effort": "low"},
        max_output_tokens=400,
    )
    
    print(response.model_dump_json(indent=2))


def test_reasoning_with_vllm():
    """
    Observation: reasoning content is returned in the reasoning output object.
    """

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="random",
    )

    print("Testing reasoning with vllm")

    response = client.responses.create(
        model="Qwen/Qwen3-0.6B",
        input=INPUT,
        reasoning={"effort": "low"},
        max_output_tokens=200,
    )
    
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    openai_token = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_token:
        print("OPENAI_API_KEY is not set in env")
        exit(1)

    test_reasoning_with_lls()

    test_reasoning_with_ollama()

    test_reasoning_with_openai()

    test_reasoning_with_vllm()
