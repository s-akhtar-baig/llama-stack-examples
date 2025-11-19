# llama-stack-examples
Provides a guide for llama stack setup and example applications.

## Installation and Configuration

**i. Install Inference Provider**

Refer to the [Ollama website](https://ollama.com/download) for instructions on installing Ollama. Once installed, download the required inference and safety models and start the Ollama service.

```
ollama pull llama3.2:3b
ollama pull llama-guard3:1b
ollama serve
```

**ii. Setup your Virtual Environment**

Install [uv](https://docs.astral.sh/uv) to setup a Python virtual environment. Next, setup your virtual environment as shown below.

```
uv sync
source .venv/bin/activate
```

**iii. Run Llama Stack**

We have provided a custom run.yaml file to specify the required providers. Use the following command to run the Llama Stack with the custom configuration file.

```
uv run llama stack run run.yaml --image-type venv
```

**iv. Validate Setup**

Open a new terminal and navigate to the llama-stack-examples directory. Activate your existing virtual environment and use the CLI tool to test your setup.

```
source .venv/bin/activate
uv run llama-stack-client configure --endpoint http://localhost:8321 --api-key none
```

**v. Run the Sample Code**

We have provided examples to showcase integration of crewai and guardrails with Llama Stack. Use the commands provided under the relevant directory to test the desired script(s):

- [crewai](crewai/README.md): sample code to use crewai with a local instance of Llama Stack
- [responses](responses/README.md): sample code to explore Responses API behavior in Llama Stack and OpenAI
- [safety](safety/README.md): sample code for content moderation on user input and the target LLM output
