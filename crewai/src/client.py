#
# This file provides a sample client that connects to a local Llama Stack instance
#

from crewai.llm import LLM

llm = LLM(
    model="meta_llama/ollama/llama3.2:3b",
    base_url="http://localhost:8321/v1/openai/v1",
    api_key="some-key" # key not required
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a two sentence poem about CrewAI."}
]

response = llm.call(messages)

print(response)
