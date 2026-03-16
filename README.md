# ai-gateway

Two independent Bedrock gateway services:

| Service | Port | API format | Use with |
|---|---|---|---|
| `bedrock-proxy` | `4002` | Anthropic Messages API | Claude Code |
| `litellm` | `4000` | OpenAI-compatible API | Cursor, Continue, etc. |

Both call AWS Bedrock directly — they do not route through each other.

## bedrock-proxy

Accepts [Anthropic Messages API](https://docs.anthropic.com/en/api/messages) requests, translates them to Bedrock Converse API format, and returns Anthropic-format responses.

1. Receives requests on `POST /v1/messages`
2. Translates to Bedrock Converse API format
3. Calls the model via `boto3`
4. Returns Anthropic-format response

## Supported models

| Alias | Model |
|---|---|
| `deepseek-r1` | DeepSeek R1 |
| `deepseek-v3` / `deepseek-v3.2` | DeepSeek V3 |
| `qwen3-32b`, `qwen3-coder`, `qwen3-80b`, `qwen3-vl` | Qwen 3 family |
| `claude-*` | Anthropic Claude (all versions) |
| `nova-premier`, `nova-pro`, `nova-lite`, `nova-micro` | Amazon Nova |
| `minimax-m2` / `minimax-m2.1` | MiniMax |
| `kimi-k2` | Kimi K2 |
| `mistral-large-3`, `magistral-small`, `devstral-2`, `pixtral-large` | Mistral |

## litellm

OpenAI-compatible gateway backed by LiteLLM, calling Bedrock directly. Use this with any tool that speaks the OpenAI API (Cursor, Continue, Open WebUI, etc.).

Config: `litellm_config.yaml`

---

## Requirements

- Docker
- AWS credentials with Bedrock model access

## Running with Docker Compose

```bash
# create a .env file with your AWS credentials
cp .env.example .env

docker compose up -d
```

Or run a single service:

```bash
docker compose up -d bedrock-proxy   # port 4002
docker compose up -d litellm         # port 4000
```

## Running bedrock-proxy without Docker

```bash
pip install fastapi uvicorn boto3
python bedrock_proxy.py
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `PROXY_PORT` | `4002` | Port to listen on |
| `AWS_REGION` | `us-east-1` | AWS region for Bedrock |

## Using with Claude Code

```bash
export CLAUDE_CODE_USE_BEDROCK=""
export ANTHROPIC_BASE_URL=http://localhost:4002
export ANTHROPIC_API_KEY=any-value
export ANTHROPIC_MODEL=deepseek-r1
export ANTHROPIC_SMALL_FAST_MODEL=deepseek-r1
```

Then run `claude` as normal.

## Using litellm with OpenAI-compatible tools

```
base_url: http://localhost:4000
api_key: sk-litellm
```

## bedrock-proxy endpoints

- `POST /v1/messages` — main inference endpoint
- `GET /v1/models` — list available model aliases
- `GET /health` — health check
