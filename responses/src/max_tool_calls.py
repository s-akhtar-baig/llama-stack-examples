"""
Examples used to explore OpenAI Responses API behavior with max_tool_call.
"""

import json
import os
from openai import OpenAI

def test_function_tools():
    """
    Observation: max_tool_calls does not impact function tools.
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
        {
            "type": "function",
            "name": "get_time",
            "description": "Get current time for a specific location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name (e.g., 'New York', 'London')",
                    },
                },
                "required": ["location"],
            },
        },
        {
            "type": "function",
            "name": "calculate_distance",
            "description": "Calculate distance between two locations",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_location": {
                        "type": "string",
                        "description": "Starting city name",
                    },
                    "to_location": {
                        "type": "string",
                        "description": "Destination city name",
                    },
                },
                "required": ["from_location", "to_location"],
            },
        },
    ]

    # Create a user query that requires multiple tool calls
    input_messages = [
        {
            "role": "user",
            "content": "I'm planning to travel from New York to Paris. Can you tell me the weather in both cities, the current time, and how far apart they are?"
        }
    ]

    print("Testing function tool calling with max_tool_calls=1")

    response = client.responses.create(
        model="gpt-4o",
        input=input_messages,
        tools=tools,
        max_tool_calls=1,
    )

    # Count tool calls with max_tool_calls set to 1
    func_calls = [output for output in response.output if output.type == "function_call"]
    print(f"Total function tool calls: {len(func_calls)}")

    # print(response.model_dump_json(indent=2))

def test_builtin_tools():
    """
    Observation: max_tool_calls impacts number of calls made to web_search.
    """

    client = OpenAI()

    try:
        print("Testing built-in tool calling with max_tool_calls=1")

        response = client.responses.create(
            model="gpt-4o",
            input="Search for a positive news story and the top rated news story from today. You MUST make two separate web search tool calls.",
            tools=[
                {"type": "web_search"},
            ],
            max_tool_calls=1,
        )

        # Count tool calls with max_tool_calls set to 1
        builtin_calls = [output for output in response.output if output.type == "web_search_call"]
        print(f"Total web search calls: {len(builtin_calls)}")

        # print(response.model_dump_json(indent=2))

    except Exception as e:
        print(f"Error: {e}")
        exit(1)

def test_mcp_tools():
    """
    Observation: 
    1. max_tool_calls impacts total number of calls to the mcp tools.
    2. max_tool_calls applies to multiple inference calls. In this experiment, the first inference call lists
       branches and the second lists commits.
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

    input = "You MUST first list branches in the llama-stack-examples repository for the user s-akhtar-baig, then list five most recent commits in any of the list of branches."

    try:
        print("Testing mcp tool calling with max_tool_calls=1")

        response = client.responses.create(
            model="gpt-4o",
            input=input,
            tools=tools,
            max_tool_calls=1,
        )

        # Count tool calls with max_tool_calls set to 1
        mcp_calls = [output for output in response.output if output.type == "mcp_call"]
        print(f"Total mcp tool calls: {len(mcp_calls)}")

        # print(response.model_dump_json(indent=2))

        print("Testing mcp tool calling with max_tool_calls=5")

        response2 = client.responses.create(
            model="gpt-4o",
            input=input,
            tools=tools,
            max_tool_calls=5,
        )

        # Count tool calls with max_tool_calls set to 5
        mcp_calls_2 = [output for output in response2.output if output.type == "mcp_call"]
        print(f"Total mcp tool calls: {len(mcp_calls_2)}")

        # print(response2.model_dump_json(indent=2))

    except Exception as e:
        print(f"Error: {e}")
        exit(1)
 
if __name__ == "__main__":
    openai_token = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_token:
        print("OPENAI_API_KEY is not set in env")
        exit(1)

    test_function_tools()

    test_builtin_tools()

    test_mcp_tools()
