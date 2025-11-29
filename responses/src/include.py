"""
Examples used to explore OpenAI Responses API behavior with the include parameter.

This module specifically focuses on testing the include["message.output_text.logprobs"] parameter,
which returns log probabilities for the generated output text.
"""

import json
import os
from openai import OpenAI


def test_basic_logprobs():
    """
    Test basic logprobs inclusion in the response.

    Observation: When include=["message.output_text.logprobs"] is set,
    the response includes log probabilities for each token..
    """
    client = OpenAI()

    print("\n=== Testing basic logprobs inclusion ===")

    input_messages = [
        {
            "role": "user",
            "content": "Write a short haiku about the ocean using five words."
        }
    ]

    # Without logprobs
    print("\n1. Response WITHOUT logprobs:")
    response_no_logprobs = client.responses.create(
        model="gpt-4o",
        input=input_messages,
    )

    print(response_no_logprobs.model_dump_json(indent=2))

    # With logprobs
    print("\n2. Response WITH logprobs:")
    response_with_logprobs = client.responses.create(
        model="gpt-4o",
        input=input_messages,
        include=["message.output_text.logprobs"],
    )

    print(response_with_logprobs.model_dump_json(indent=2))

    # With logprobs and stream set to True
    print("\n3. Response stream WITH logprobs:")
    response_with_logprobs_2 = client.responses.create(
        model="gpt-4o",
        input=input_messages,
        include=["message.output_text.logprobs"],
        stream=True,
    )

    for chunk in response_with_logprobs_2:
        print(chunk.model_dump_json(indent=2))


def test_logprobs_with_function_tool_calls():
    """
    Test logprobs when the response includes function tool calls.

    Observation: Logprobs aren't included in the model's response
    for function tools.
    """
    client = OpenAI()

    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get current weather information for a specific location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name (e.g., 'New York', 'London')",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["fahrenheit", "celsius"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["location"],
            },
        },
    ]

    print("\n=== Testing logprobs with function tools ===")

    response_w_function_tools = client.responses.create(
        model="gpt-4o",
        input="What's the weather like in Paris?",
        tools=tools,
        include=["message.output_text.logprobs"],
    )

    print(response_w_function_tools.model_dump_json(indent=2))


def test_logprobs_with_builtin_tools():
    """
    Test logprobs when the response includes built-in tool calls.

    Observation: Logprobs aren't included in the model's response
    for built-in tools.
    """
    client = OpenAI()

    tools = [
        { "type": "web_search" },
    ]

    print("\n=== Testing logprobs with built-in tools ===")

    response_w_builtin_tools = client.responses.create(
        model="gpt-4o",
        input="Search for a positive news story from today.",
        tools=tools,
        include=["message.output_text.logprobs"],
    )

    print(response_w_builtin_tools.model_dump_json(indent=2))


def test_logprobs_with_mcp_tools():
    """
    Test logprobs when the response includes mcp tool calls.

    Observation: Logprobs are included in the model's response
    for mcp tools.
    """
    client = OpenAI()

    gh_token = os.getenv("GITHUB_TOKEN", "").strip()
    if not gh_token:
        print("GITHUB_TOKEN not set in env")
        exit(1)

    tools = [
        {
            "type": "mcp",
            "server_label": "github",
            "server_url": "https://api.githubcopilot.com/mcp/x/repos/readonly",
            "require_approval": "never",
            "headers": {
                "Authorization": f"Bearer {gh_token}"
            }
        }
    ]

    print("\n=== Testing logprobs with mcp tools ===")

    response_w_mcp_tools = client.responses.create(
        model="gpt-4o",
        input="List branches in the llama-stack-examples repository for the user s-akhtar-baig.",
        tools=tools,
        include=["message.output_text.logprobs"],
    )

    print(response_w_mcp_tools.model_dump_json(indent=2))


def test_logprobs_with_builtin_and_mcp_tools():
    """
    Test logprobs when the response includes built-in and mcp tool calls.

    Observation:
    1. Logprobs are included in the model's response when web search call is made before the mcp tool and there are two assistant messages.
    1. Logprobs aren't included in the model's response when mcp tool is made called before web search and there is one assistant message.
    """
    client = OpenAI()

    gh_token = os.getenv("GITHUB_TOKEN", "").strip()
    if not gh_token:
        print("GITHUB_TOKEN not set in env")
        exit(1)

    tools = [
        {
            "type": "mcp",
            "server_label": "github",
            "server_url": "https://api.githubcopilot.com/mcp/x/repos/readonly",
            "require_approval": "never",
            "headers": {
                "Authorization": f"Bearer {gh_token}"
            }
        },
        { 
            "type": "web_search" 
        },
    ]

    print("\n=== Testing logprobs with built-in first and mcp tool second ===")

    response_w_multiple_tools = client.responses.create(
        model="gpt-4o",
        input="Search for a positive news story from today and list branches in the llama-stack-examples repository for the user s-akhtar-baig.",
        tools=tools,
        include=["message.output_text.logprobs"],
    )

    print(response_w_multiple_tools.model_dump_json(indent=2))

    print("\n=== Testing logprobs with mcp first and built-in second ===")

    response_w_multiple_tools_2 = client.responses.create(
        model="gpt-4o",
        input="List branches in the llama-stack-examples repository for the user s-akhtar-baig and search for a positive news story from today.",
        tools=tools,
        include=["message.output_text.logprobs"],
    )

    print(response_w_multiple_tools_2.model_dump_json(indent=2))


def test_logprobs_with_function_and_mcp_tools():
    """
    Test logprobs when the response includes function and mcp tool calls.

    Observation: Logprobs aren't included in the model's response
    for function and mcp tools.
    """
    client = OpenAI()

    gh_token = os.getenv("GITHUB_TOKEN", "").strip()
    if not gh_token:
        print("GITHUB_TOKEN not set in env")
        exit(1)

    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get current weather information for a specific location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name (e.g., 'New York', 'London')",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["fahrenheit", "celsius"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["location"],
            },
        },
        {
            "type": "mcp",
            "server_label": "github",
            "server_url": "https://api.githubcopilot.com/mcp/x/repos/readonly",
            "require_approval": "never",
            "headers": {
                "Authorization": f"Bearer {gh_token}"
            }
        },
    ]

    print("\n=== Testing logprobs with function and mcp tools ===")

    response_w_multiple_tools = client.responses.create(
        model="gpt-4o",
        input="What's the weather like in Paris and list branches in the llama-stack-examples repository for the user s-akhtar-baig.",
        tools=tools,
        include=["message.output_text.logprobs"],
    )

    print(response_w_multiple_tools.model_dump_json(indent=2))


if __name__ == "__main__":
    openai_token = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_token:
        print("OPENAI_API_KEY is not set in env")
        exit(1)

    test_basic_logprobs()

    test_logprobs_with_function_tool_calls()

    test_logprobs_with_builtin_tools()

    test_logprobs_with_mcp_tools()

    test_logprobs_with_builtin_and_mcp_tools()

    test_logprobs_with_function_and_mcp_tools()
