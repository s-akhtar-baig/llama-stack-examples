#
# This file provides a sample client that connects to a local Llama Stack instance
#

from crewai import LLM, Agent, Task, Crew

llm = LLM(
    model="meta_llama/ollama/llama3.2:3b",
    base_url="http://localhost:8321/v1/openai/v1",
    api_key="some-key" # key not required
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a two sentence poem about CrewAI."}
]

print("Testing llm call to llama3.2:3b:")
response = llm.call(messages)
print(response)
print("-" * 10)

researcher = Agent(
    role="About LLM",
    goal="You know everything about LLM.",
    backstory="""You are a master at LLMs and their safety issues.""",
    llm=llm,
)

search = Task(
    description="Answer the following questions about LLMs: {question}",
    expected_output="An answer to the question.",
    agent=researcher,
)

crew = Crew(agents=[researcher], tasks=[search])

result = crew.kickoff(
    inputs={"question": "What is the latest on LLM and guardrails?"}
)

print("Testing crewai agent and task execution:")
print(result)
