# Responses and Llama Stack

The example scripts provided under the src directory will help explore Responses API behavior in OpenAI. It is still a work in progress.

## Installation and Configuration

### Llama Stack

Refer to the [Installation and Configuration](../README.md#installation-and-configuration) section for instructions on running a Llama Stack instance with the required providers.

### GitHub Setup

#### 1. Generate a Fine-Grained Access Token

1. Go to GitHub Settings → Developer settings → Personal access tokens → [Fine-grained tokens](https://github.com/settings/personal-access-tokens)
2. To get a detailed description of the required fields and steps, visit [Manage PAT](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token)
3. Ensure that the Repository access is set to "Public repositories"
4. Click "Generate token"
5. **Copy the token** and set it as an environment variable: `export GITHUB_TOKEN=<your_gh_token>`

#### 2. Setup GitHub MCP Server

The provided scripts use the remote MCP server hosted by GitHub. Users must obtain a fine-grained token (as described in step 1) to grant required permissions to the LLMs.

### Required Environment Variables

Set tokens associated to your OpenAI account and GitHub account (refer to steps described under GitHub Setup section).

```
export OPENAI_API_KEY=<your_openai_key>
export GITHUB_TOKEN=<your_gh_token>
```

## Run the Sample Code

Navigate to the responses directory and use the following command to test the provided scripts:

```
uv run python src/max_tool_calls.py 
```

```
uv run python src/include.py
```
