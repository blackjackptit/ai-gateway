# ai-gateway

A local proxy that accepts [Anthropic Messages API](https://docs.anthropic.com/en/api/messages) requests and forwards them to AWS Bedrock via the Converse API. Lets you use tools like Claude Code with non-Anthropic models hosted on Bedrock.

## How it works

1. Receives requests in Anthropic Messages API format on `POST /v1/messages`
2. Translates them to Bedrock Converse API format
3. Calls the model via `boto3`
4. Returns the response in Anthropic Messages API format

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

## Requirements

- Python 3.10+
- AWS credentials configured (e.g. via `~/.aws/credentials` or environment variables)
- Bedrock model access enabled in your AWS account

```bash
pip install fastapi uvicorn boto3
```

## Usage

```bash
python bedrock_proxy.py
```

The proxy starts on port `4002` by default.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `PROXY_PORT` | `4002` | Port to listen on |
| `AWS_REGION` | `us-east-1` | AWS region for Bedrock |

### Using with Claude Code

```bash
export CLAUDE_CODE_USE_BEDROCK=""
export ANTHROPIC_BASE_URL=http://localhost:4002
export ANTHROPIC_API_KEY=any-value
export ANTHROPIC_MODEL=deepseek-r1
export ANTHROPIC_SMALL_FAST_MODEL=deepseek-r1
```

Then run `claude` as normal.

## Endpoints

- `POST /v1/messages` — main inference endpoint
- `GET /v1/models` — list available model aliases
- `GET /health` — health check
