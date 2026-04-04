"""
Minimal Anthropic-to-Bedrock proxy.

Accepts Anthropic Messages API requests from Claude Code,
strips unsupported params, calls models on Bedrock via boto3,
returns Anthropic-format responses.

Usage:
  python bedrock_proxy.py

Then set:
  CLAUDE_CODE_USE_BEDROCK=""
  ANTHROPIC_BASE_URL=http://localhost:4002
  ANTHROPIC_API_KEY=any-value
  ANTHROPIC_MODEL=deepseek-r1     
  ANTHROPIC_SMALL_FAST_MODEL=deepseek-r1
"""

import asyncio
import json
import uuid
import time
import os
import boto3
from botocore.config import Config
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()
_executor = ThreadPoolExecutor()


@app.middleware("http")
async def add_github_headers(request, call_next):
    # Don't log timing for streaming - it skews and blocks until stream ends
    # Just add the header
    response = await call_next(request)
    response.headers["x-github-request-id"] = str(uuid.uuid4())
    return response

# Model ID mapping
MODEL_MAP = {
    # DeepSeek
    "deepseek-r1":   "us.deepseek.r1-v1:0",
    "deepseek-v3":   "deepseek.v3.2",
    "deepseek-v3.2": "deepseek.v3.2",
    # Qwen3 (32K context)
    "qwen3-32b":         "qwen.qwen3-32b-v1:0",
    "qwen3-coder":           "qwen.qwen3-coder-30b-a3b-v1:0",
    "qwen3-coder-next":      "qwen.qwen3-coder-next",
    "qwen3-80b":             "qwen.qwen3-next-80b-a3b",
    "qwen3-vl":              "qwen.qwen3-vl-235b-a22b",
    # Qwen full-id aliases Claude Code may send
    "qwen.qwen3-coder":      "qwen.qwen3-coder-30b-a3b-v1:0",
    "qwen.qwen3-32b":        "qwen.qwen3-32b-v1:0",
    "qwen.qwen3-80b":        "qwen.qwen3-next-80b-a3b",
    "qwen.qwen3-vl":         "qwen.qwen3-vl-235b-a22b",
    # Anthropic Claude (200K context)
    "claude-haiku-4-5":       "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-haiku-4.5":       "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-sonnet-4-5":      "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-sonnet-4.5":      "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-sonnet-4-6":      "us.anthropic.claude-sonnet-4-6",
    "claude-sonnet-4.6":      "us.anthropic.claude-sonnet-4-6",
    "claude-opus-4-5":        "us.anthropic.claude-opus-4-5-20251101-v1:0",
    "claude-opus-4.5":        "us.anthropic.claude-opus-4-5-20251101-v1:0",
    "claude-opus-4-6":        "us.anthropic.claude-opus-4-6-v1",
    "claude-opus-4.6":        "us.anthropic.claude-opus-4-6-v1",
    "claude-opus-4.6-fast":   "us.anthropic.claude-opus-4-6-v1",
    "claude-opus-4":          "us.anthropic.claude-opus-4-20250514-v1:0",
    "claude-sonnet-4":        "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-sonnet-3-7":      "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "claude-sonnet-3-5-v2":   "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-haiku-3-5":       "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "claude-sonnet-3-5":      "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    "claude-opus-3":          "us.anthropic.claude-3-opus-20240229-v1:0",
    # Passthrough aliases (full model IDs Claude Code may send directly)
    "claude-3-5-sonnet-20241022":   "us.anthropic.claude-sonnet-4-5-20250929-v1:0",  # legacy → sonnet-4-5
    "claude-3-5-sonnet-20240620":   "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    "claude-3-5-haiku-20241022":    "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "claude-3-7-sonnet-20250219":   "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "claude-3-opus-20240229":       "us.anthropic.claude-3-opus-20240229-v1:0",
    "claude-sonnet-4-20250514":     "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-opus-4-20250514":       "us.anthropic.claude-opus-4-20250514-v1:0",
    "claude-haiku-4-5-20251001":    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-sonnet-4-5-20250929":   "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-opus-4-5-20251101":     "us.anthropic.claude-opus-4-5-20251101-v1:0",
    # Amazon Nova
    "nova-premier":  "us.amazon.nova-premier-v1:0",   # 1M context
    "nova-pro":      "us.amazon.nova-pro-v1:0",        # 300K context
    "nova-lite":     "us.amazon.nova-lite-v1:0",       # 300K context
    "nova-2-lite":   "us.amazon.nova-2-lite-v1:0",     # 300K context
    "nova-micro":    "us.amazon.nova-micro-v1:0",      # 128K context
    # Meta Llama 4 — geo-restricted, not available in this region
    # "llama4-scout":    "us.meta.llama4-scout-17b-instruct-v1:0",
    # "llama4-maverick": "us.meta.llama4-maverick-17b-instruct-v1:0",
    # MiniMax
    "minimax-m2":   "minimax.minimax-m2",    # 1M context
    "minimax-m2.1": "minimax.minimax-m2.1",  # 1M context
    "minimax-m2.5": "minimax.minimax-m2.5",  # 1M context
    # GLM (Zhipu AI)
    "glm-5":        "zhipuai.glm-5",         # 128K context
    "glm-5-plus":   "zhipuai.glm-5-plus",    # 128K context
    # Kimi
    "kimi-k2":         "moonshotai.kimi-k2.5",       # 128K context
    "kimi-k2-thinking": "moonshot.kimi-k2-thinking", # 128K context
    # Mistral large context
    "mistral-large-3":  "mistral.mistral-large-3-675b-instruct",
    "magistral-small":  "mistral.magistral-small-2509",
    "devstral-2":       "mistral.devstral-2-123b",
    "pixtral-large":    "us.mistral.pixtral-large-2502-v1:0",
}

# Models that support Converse API toolConfig (native tool calling)
TOOLS_SUPPORTED = {
    # Qwen3
    "qwen3-32b", "qwen3-coder", "qwen3-coder-next", "qwen3-80b", "qwen3-vl",
    "qwen.qwen3-coder", "qwen.qwen3-32b", "qwen.qwen3-80b", "qwen.qwen3-vl",
    # Claude (all versions)
    "claude-haiku-4-5", "claude-haiku-4.5",
    "claude-sonnet-4-5", "claude-sonnet-4.5",
    "claude-sonnet-4-6", "claude-sonnet-4.6",
    "claude-opus-4-5", "claude-opus-4.5",
    "claude-opus-4-6", "claude-opus-4.6", "claude-opus-4.6-fast",
    "claude-opus-4", "claude-sonnet-4",
    "claude-sonnet-3-7", "claude-sonnet-3-5-v2", "claude-haiku-3-5",
    "claude-sonnet-3-5", "claude-opus-3",
    # Full ID aliases
    "claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620", "claude-3-5-haiku-20241022",
    "claude-3-7-sonnet-20250219", "claude-3-opus-20240229", "claude-sonnet-4-20250514",
    "claude-opus-4-20250514", "claude-haiku-4-5-20251001", "claude-sonnet-4-5-20250929",
    "claude-opus-4-5-20251101",
    # Amazon Nova
    "nova-premier", "nova-pro", "nova-lite", "nova-2-lite", "nova-micro",
    # Meta Llama 4 — geo-restricted
    # "llama4-scout", "llama4-maverick",
    # MiniMax
    "minimax-m2", "minimax-m2.1", "minimax-m2.5",
    # GLM (Zhipu AI)
    "glm-5", "glm-5-plus",
    # Mistral
    "mistral-large-3", "magistral-small", "devstral-2", "pixtral-large",
    # Kimi — outputs raw text markup, NOT added here (not supported)
    # "kimi-k2", "kimi-k2-thinking",
}

# Models requiring strict tool-block separation (no mixing text + toolUse/toolResult in same turn)
# Claude and Nova models support mixed content; third-party models typically do not.
STRICT_TOOL_SEPARATION = {
    "qwen3-32b", "qwen3-coder", "qwen3-coder-next", "qwen3-80b", "qwen3-vl",
    "qwen.qwen3-coder", "qwen.qwen3-32b", "qwen.qwen3-80b", "qwen.qwen3-vl",
    "minimax-m2", "minimax-m2.1", "minimax-m2.5",
    "glm-5", "glm-5-plus",
    "mistral-large-3", "magistral-small", "devstral-2", "pixtral-large",
}

# Total context window length for models (in tokens)
MODEL_CONTEXT_LENGTHS = {
    # MiniMax models - 196608 tokens (192K)
    "minimax-m2":   196608,
    "minimax-m2.1": 196608,
    "minimax-m2.5": 196608,
    # GLM models - 131072 tokens (128K)
    "glm-5":        131072,
    "glm-5-plus":   131072,
    # Kimi models - 131072 tokens (128K)
    "kimi-k2":         131072,
    "kimi-k2-thinking": 131072,
    # Qwen3 models - 32768 tokens (32K)
    "qwen3-32b":        32768,
    "qwen3-coder":      32768,
    "qwen3-coder-next": 32768,
    "qwen3-80b":        32768,
    "qwen3-vl":         32768,
    "qwen.qwen3-coder": 32768,
    "qwen.qwen3-32b":   32768,
    "qwen.qwen3-80b":   32768,
    "qwen.qwen3-vl":    32768,
}

# Hard cap on output tokens per model
MODEL_MAX_TOKENS = {
    # Qwen3 (32K context window, Bedrock max output 8192)
    "qwen3-32b":        8192,
    "qwen3-coder":      8192,
    "qwen3-coder-next": 8192,
    "qwen3-80b":        8192,
    "qwen3-vl":         8192,
    "qwen.qwen3-coder": 8192,
    "qwen.qwen3-32b":   8192,
    "qwen.qwen3-80b":   8192,
    "qwen.qwen3-vl":    8192,
    # Amazon Nova output limits per Bedrock docs
    "nova-pro":     5120,
    "nova-lite":    5120,
    "nova-2-lite":  5120,
    "nova-micro":   5120,
    "nova-premier": 10240,
    # MiniMax models (196K context, reduced to 4K to leave room for large inputs)
    "minimax-m2":   4096,
    "minimax-m2.1": 4096,
    "minimax-m2.5": 4096,
    # GLM models (128K context, conservative limit)
    "glm-5":        4096,
    "glm-5-plus":   4096,
}

# Max system prompt characters — leave headroom for messages within the context window
MODEL_MAX_SYSTEM_CHARS = {
    # Qwen3 32K context: ~24K chars (~6K tokens) for system, rest for messages
    "qwen3-32b":        24000,
    "qwen3-coder":      24000,
    "qwen3-coder-next": 24000,
    "qwen.qwen3-32b":   24000,
    "qwen.qwen3-coder": 24000,
}

# Model-specific read timeouts (in seconds) for slower/large-context models
MODEL_READ_TIMEOUTS = {
    # MiniMax 1M context models may take longer
    "minimax-m2":       420,  # 7 minutes
    "minimax-m2.1":     420,  # 7 minutes
    "minimax-m2.5":     420,  # 7 minutes
    # DeepSeek reasoning models
    "deepseek-r1":      360,  # 6 minutes
    # GLM models (may be slower with tool calls)
    "glm-5":            360,  # 6 minutes
    "glm-5-plus":       360,  # 6 minutes
    # Kimi reasoning models
    "kimi-k2-thinking": 360,  # 6 minutes
    # Default for other models is 300s (5 minutes) set in get_bedrock_client
}

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

_bedrock_client: boto3.client = None
_model_specific_clients: dict = {}

def get_bedrock_client(model_alias: str = None):
    """Get a Bedrock client with appropriate timeout for the model."""
    # If no specific model requested, return default client
    if model_alias is None:
        global _bedrock_client
        if _bedrock_client is None:
            config = Config(
                read_timeout=300,      # 5 minutes default
                connect_timeout=10,    # 10 seconds to establish connection
                retries={
                    'max_attempts': 3,
                    'mode': 'adaptive'  # Exponential backoff with jitter
                }
            )
            _bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name=AWS_REGION,
                config=config
            )
        return _bedrock_client

    # Check if model needs custom timeout
    custom_timeout = MODEL_READ_TIMEOUTS.get(model_alias)
    if custom_timeout is None:
        # Use default client for models without special timeout needs
        return get_bedrock_client(None)

    # Create or retrieve model-specific client
    if model_alias not in _model_specific_clients:
        config = Config(
            read_timeout=custom_timeout,
            connect_timeout=10,
            retries={
                'max_attempts': 3,
                'mode': 'adaptive'
            }
        )
        _model_specific_clients[model_alias] = boto3.client(
            "bedrock-runtime",
            region_name=AWS_REGION,
            config=config
        )

    return _model_specific_clients[model_alias]


DEFAULT_MODEL = "claude-haiku-4-5"

def anthropic_to_converse(body: dict) -> tuple[str, dict]:
    """Convert Anthropic Messages API body to Bedrock Converse API format."""
    model_alias = body.get("model", DEFAULT_MODEL)
    if model_alias not in MODEL_MAP:
        print(f"[WARN] Unknown model '{model_alias}', falling back to '{DEFAULT_MODEL}'")
        model_alias = DEFAULT_MODEL
    bedrock_model_id = MODEL_MAP[model_alias]

    # Build messages — Converse only supports user/assistant roles
    messages = []
    for msg in body.get("messages", []):
        role = msg["role"]
        content = msg["content"]
        if role not in ("user", "assistant"):
            continue
        if isinstance(content, str):
            # Skip empty strings
            if content.strip():
                converse_content = [{"text": content}]
            else:
                converse_content = []
        else:
            # Handle content blocks
            converse_content = []
            for block in content:
                if isinstance(block, str):
                    # Skip empty strings
                    if block.strip():
                        converse_content.append({"text": block})
                elif isinstance(block, dict):
                    btype = block.get("type")
                    if btype == "text":
                        # Skip empty text blocks
                        text_content = block.get("text", "")
                        if text_content.strip():
                            converse_content.append({"text": text_content})
                    elif btype == "tool_use":
                        if model_alias in TOOLS_SUPPORTED:
                            converse_content.append({
                                "toolUse": {
                                    "toolUseId": block["id"],
                                    "name": block["name"],
                                    "input": block.get("input", {}),
                                }
                            })
                        else:
                            # Flatten to text for models without native tool support
                            converse_content.append({
                                "text": f"[Tool call: {block['name']}({json.dumps(block.get('input', {}))})]"
                            })
                    elif btype == "tool_result":
                        if model_alias in TOOLS_SUPPORTED:
                            result_content = block.get("content", "")
                            if isinstance(result_content, str):
                                # Skip empty result strings
                                tc = [{"text": result_content}] if result_content.strip() else [{"text": "[empty result]"}]
                            elif isinstance(result_content, list):
                                tc = [
                                    {"text": b["text"]}
                                    for b in result_content
                                    if isinstance(b, dict) and b.get("type") == "text" and b.get("text", "").strip()
                                ]
                                # If all text blocks were empty, add placeholder
                                if not tc:
                                    tc = [{"text": "[empty result]"}]
                            else:
                                tc = [{"text": str(result_content)}]
                            converse_content.append({
                                "toolResult": {
                                    "toolUseId": block["tool_use_id"],
                                    "content": tc,
                                }
                            })
                        else:
                            # Flatten to text for models without native tool support
                            result_content = block.get("content", "")
                            if isinstance(result_content, list):
                                result_content = " ".join(
                                    b.get("text", "") for b in result_content
                                    if isinstance(b, dict)
                                )
                            converse_content.append({
                                "text": f"[Tool result: {result_content or '[empty]'}]"
                            })
            if not converse_content:
                converse_content = [{"text": "[empty]"}]

        # Bedrock forbids mixing text blocks with toolUse/toolResult in the same turn
        # for models that require strict separation (non-Claude, non-Nova).
        if model_alias in STRICT_TOOL_SEPARATION:
            has_tool_blocks = any("toolUse" in b or "toolResult" in b for b in converse_content)
            if has_tool_blocks:
                converse_content = [b for b in converse_content if "toolUse" in b or "toolResult" in b]
                if not converse_content:
                    converse_content = [{"text": "[empty]"}]

        messages.append({"role": role, "content": converse_content})

    converse_body = {"messages": messages}

    # System prompt
    system = body.get("system")
    if system:
        if isinstance(system, str):
            system_text = system
        elif isinstance(system, list):
            texts = [b["text"] for b in system if isinstance(b, dict) and b.get("type") == "text"]
            system_text = "\n\n".join(texts) if texts else ""
        else:
            system_text = ""
        if system_text:
            max_chars = MODEL_MAX_SYSTEM_CHARS.get(model_alias)
            if max_chars and len(system_text) > max_chars:
                system_text = system_text[:max_chars] + "\n\n[System prompt truncated]"
            converse_body["system"] = [{"text": system_text}]

    # Tools → toolConfig (only for models that support native tool calling)
    tools = body.get("tools")
    if tools and model_alias in TOOLS_SUPPORTED:
        tool_specs = []
        for t in tools:
            spec = {"name": t["name"]}
            if t.get("description"):
                spec["description"] = t["description"]
            # inputSchema is required by Bedrock — default to empty object if absent
            spec["inputSchema"] = {"json": t["input_schema"]} if t.get("input_schema") else {"json": {"type": "object", "properties": {}}}
            tool_specs.append({"toolSpec": spec})
        converse_body["toolConfig"] = {"tools": tool_specs}

    inference_config = {}
    if "max_tokens" in body:
        requested = body["max_tokens"]
        cap = MODEL_MAX_TOKENS.get(model_alias)
        inference_config["maxTokens"] = min(requested, cap) if cap else requested
    if "temperature" in body:
        inference_config["temperature"] = body["temperature"]
    if "top_p" in body and "temperature" not in body:
        # Claude/Bedrock rejects requests with both temperature and top_p set
        inference_config["topP"] = body["top_p"]
    if "stop_sequences" in body:
        inference_config["stopSequences"] = body["stop_sequences"]

    if inference_config:
        converse_body["inferenceConfig"] = inference_config

    return bedrock_model_id, converse_body


def converse_to_anthropic(model_alias: str, converse_resp: dict) -> dict:
    """Convert Bedrock Converse response to Anthropic Messages API format."""
    output = converse_resp.get("output", {})
    message = output.get("message", {})
    content_blocks = message.get("content", [])

    anthropic_content = []
    for block in content_blocks:
        if "text" in block:
            anthropic_content.append({"type": "text", "text": block["text"]})
        elif "toolUse" in block:
            tu = block["toolUse"]
            anthropic_content.append({
                "type": "tool_use",
                "id": tu["toolUseId"],
                "name": tu["name"],
                "input": tu.get("input", {}),
            })

    stop_reason_map = {
        "end_turn":    "end_turn",
        "stop_sequence": "stop_sequence",
        "max_tokens":  "max_tokens",
        "tool_use":    "tool_use",
    }
    stop = converse_resp.get("stopReason", "end_turn")
    stop_reason = stop_reason_map.get(stop, "end_turn")

    usage = converse_resp.get("usage", {})
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model_alias,
        "content": anthropic_content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens":  usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
        },
    }


def _sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _call_converse_with_retry(client, bedrock_model_id: str, converse_body: dict, model_alias: str):
    """Call Bedrock converse with automatic retry on context length errors.

    Detects when input genuinely exceeds context (reported input grows with each retry)
    vs when it's a tight fit that just needs maxTokens reduced by a few tokens.
    """
    import re

    context_length = MODEL_CONTEXT_LENGTHS.get(model_alias, 200000)
    prev_input_tokens = None

    for attempt in range(3):
        try:
            return client.converse(modelId=bedrock_model_id, **converse_body)
        except Exception as e:
            error_msg = str(e)
            tl = error_msg.lower()

            is_ctx = ('context length' in tl or 'maximum input length' in tl or
                      ('input tokens' in tl and ('context' in tl or 'requested' in tl)))
            if not is_ctx:
                raise

            current_max = converse_body.get("inferenceConfig", {}).get("maxTokens", 8192)

            # Extract reported input token count
            m_actual = re.search(r'passed (\d+) input tokens', error_msg)
            reported_input = int(m_actual.group(1)) if m_actual else None

            # Detect impossible case: reported_input + maxTokens always > context
            # means the actual input alone exceeds the context window
            if prev_input_tokens is not None and reported_input is not None:
                if reported_input > prev_input_tokens:
                    print(f"[CONVERSE] Input exceeds {context_length} context: tokens grew {prev_input_tokens} -> {reported_input}")
                    raise

            prev_input_tokens = reported_input

            # Calculate new maxTokens: tight fit with small buffer
            if reported_input is not None:
                new_max = max(1024, context_length - reported_input - 2)
            else:
                m_max = re.search(r'maximum input length of (\d+)', error_msg)
                if m_max:
                    new_max = max(1024, context_length - int(m_max.group(1)) - 2)
                else:
                    new_max = max(1024, current_max // 2)

            print(f"[CONVERSE] Context overflow: reducing maxTokens {current_max} -> {new_max} (attempt {attempt+1}/3)")

            if "inferenceConfig" not in converse_body:
                converse_body["inferenceConfig"] = {}
            converse_body["inferenceConfig"]["maxTokens"] = new_max

            if attempt == 2:
                raise


def _stream_bedrock_sse(client, bedrock_model_id: str, converse_body: dict, model_alias: str):
    """Blocking generator that calls converse_stream and yields SSE strings."""
    import time as time_module
    from botocore.exceptions import ClientError
    start_time = time_module.time()
    print(f"[STREAM] Starting Bedrock converse_stream for {model_alias}")

    # Try to call converse_stream with error handling for context length issues
    import re as _sre
    context_length = MODEL_CONTEXT_LENGTHS.get(model_alias, 200000)
    prev_input = None
    for attempt in range(2):
        try:
            resp = client.converse_stream(modelId=bedrock_model_id, **converse_body)
            stream = resp.get("stream", [])
            break  # Success, exit retry loop
        except Exception as e:
            error_msg = str(e)
            tl = error_msg.lower()
            is_ctx = ('context length' in tl or 'maximum input length' in tl or
                      ('input tokens' in tl and ('context' in tl or 'requested' in tl)))
            if not is_ctx:
                raise

            current_max = converse_body.get("inferenceConfig", {}).get("maxTokens", 8192)
            m_actual = _sre.search(r'passed (\d+) input tokens', error_msg)
            reported_input = int(m_actual.group(1)) if m_actual else None

            # Detect impossible case: input exceeds context
            if prev_input is not None and reported_input is not None and reported_input > prev_input + 5:
                print(f"[STREAM] Input exceeds context: {prev_input} -> {reported_input}")
                raise
            prev_input = reported_input

            if reported_input is not None:
                new_max = max(1024, context_length - reported_input - 2)
            else:
                new_max = max(1024, current_max // 2)
            print(f"[STREAM] Context overflow: reducing maxTokens {current_max} -> {new_max}")

            if "inferenceConfig" not in converse_body:
                converse_body["inferenceConfig"] = {}
            converse_body["inferenceConfig"]["maxTokens"] = new_max

            if attempt < 1:
                continue
            else:
                raise
    else:
        # Should not reach here, but handle gracefully
        resp = client.converse_stream(modelId=bedrock_model_id, **converse_body)
        stream = resp.get("stream", [])
    print(f"[STREAM] Got stream object, iterating events...")

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant",
            "model": model_alias, "content": [], "stop_reason": None,
            "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })
    yield _sse("ping", {"type": "ping"})
    print(f"[STREAM] Sent message_start and ping")

    block_index = 0
    stop_reason = "end_turn"
    output_tokens = 0
    event_count = 0
    block_started = False  # track whether content_block_start was emitted

    for event in stream:
        event_count += 1
        event_type = list(event.keys())[0] if event else "unknown"
        if event_count <= 5:
            print(f"[STREAM] Event {event_count}: {event_type}")
        if "messageStart" in event:
            pass  # already sent

        elif "contentBlockStart" in event:
            start = event["contentBlockStart"].get("start", {})
            if "toolUse" in start:
                tu = start["toolUse"]
                cb = {"type": "tool_use", "id": tu["toolUseId"], "name": tu["name"], "input": {}}
            else:
                cb = {"type": "text", "text": ""}
            yield _sse("content_block_start", {"type": "content_block_start", "index": block_index, "content_block": cb})
            block_started = True

        elif "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta", {})
            # Bedrock may skip contentBlockStart for text blocks — emit synthetic one
            if not block_started:
                yield _sse("content_block_start", {"type": "content_block_start", "index": block_index, "content_block": {"type": "text", "text": ""}})
                block_started = True
            if "text" in delta:
                text_content = delta["text"]
                # Detect context length error returned as text content (some models e.g. minimax)
                tl = text_content.lower()
                if ('context length' in tl or 'maximum input length' in tl or
                        ('input tokens' in tl and ('context' in tl or 'requested' in tl))):
                    raise ValueError(f"[CONTEXT_OVERFLOW_IN_STREAM] {text_content[:600]}")
                yield _sse("content_block_delta", {
                    "type": "content_block_delta", "index": block_index,
                    "delta": {"type": "text_delta", "text": text_content},
                })
            elif "toolUse" in delta:
                yield _sse("content_block_delta", {
                    "type": "content_block_delta", "index": block_index,
                    "delta": {"type": "input_json_delta", "partial_json": delta["toolUse"].get("input", "")},
                })

        elif "contentBlockStop" in event:
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
            block_index += 1
            block_started = False

        elif "messageStop" in event:
            br = event["messageStop"].get("stopReason", "end_turn")
            stop_reason = {"end_turn": "end_turn", "stop_sequence": "stop_sequence",
                           "max_tokens": "max_tokens", "tool_use": "tool_use"}.get(br, "end_turn")

        elif "metadata" in event:
            output_tokens = event["metadata"].get("usage", {}).get("outputTokens", 0)

    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })
    yield _sse("message_stop", {"type": "message_stop"})


async def _sse_generator(client, bedrock_model_id: str, converse_body: dict, model_alias: str):
    """Async wrapper: runs the blocking SSE generator in a thread and yields chunks.
    Automatically retries with reduced maxTokens on context length errors."""
    import re as _re
    loop = asyncio.get_event_loop()
    max_retries = 3

    for attempt in range(max_retries):
        queue: asyncio.Queue = asyncio.Queue()
        context_error: dict = {}  # Shared state to signal context length error from thread

        def run(q=queue, err=context_error):
            chunks_yielded = 0
            try:
                print(f"[SSE] Starting stream for {model_alias} -> {bedrock_model_id}")
                for chunk in _stream_bedrock_sse(client, bedrock_model_id, converse_body, model_alias):
                    loop.call_soon_threadsafe(q.put_nowait, chunk)
                    chunks_yielded += 1
                    if chunks_yielded <= 3:
                        print(f"[SSE] chunk {chunks_yielded}: {chunk[:80]}...")
                print(f"[SSE] Stream completed, yielded {chunks_yielded} chunks")
            except Exception as e:
                err_msg = str(e)
                timeout_info = f" (timeout: {MODEL_READ_TIMEOUTS.get(model_alias, 300)}s)" if "timeout" in err_msg.lower() or "read timeout" in err_msg.lower() else ""
                print(f"[SSE] ERROR Stream failed: {err_msg}{timeout_info}")
                print(f"[SSE] ERROR Model: {model_alias} -> {bedrock_model_id}")

                # Detect context length error — signal for retry
                is_ctx_error = ('context length' in err_msg.lower() or 'input length' in err_msg.lower() or
                                ('input tokens' in err_msg.lower() and 'context' in err_msg.lower()) or
                                '[context_overflow_in_stream]' in err_msg.lower())
                # Retry if: explicit marker (model returned error as text), or early failure before content
                if is_ctx_error and ('[context_overflow_in_stream]' in err_msg.lower() or chunks_yielded <= 2):
                    m_actual = _re.search(r'passed (\d+) input tokens', err_msg)
                    if m_actual:
                        err['actual_input'] = int(m_actual.group(1))
                    else:
                        m = _re.search(r'maximum input length of (\d+)', err_msg)
                        if m:
                            err['max_input'] = int(m.group(1))
                    err['retry'] = True
                    err['msg'] = err_msg
                    loop.call_soon_threadsafe(q.put_nowait, None)
                    return

                # Emit proper end-of-stream events so client doesn't hang
                if chunks_yielded == 0:
                    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
                    loop.call_soon_threadsafe(q.put_nowait, _sse("message_start", {
                        "type": "message_start",
                        "message": {"id": msg_id, "type": "message", "role": "assistant",
                                    "model": model_alias, "content": [], "stop_reason": None,
                                    "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0}},
                    }))
                    loop.call_soon_threadsafe(q.put_nowait, _sse("content_block_start", {
                        "type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}))
                    loop.call_soon_threadsafe(q.put_nowait, _sse("content_block_delta", {
                        "type": "content_block_delta", "index": 0,
                        "delta": {"type": "text_delta", "text": f"[Error: {err_msg}]"}}))
                    loop.call_soon_threadsafe(q.put_nowait, _sse("content_block_stop", {
                        "type": "content_block_stop", "index": 0}))
                loop.call_soon_threadsafe(q.put_nowait, _sse("message_delta", {
                    "type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 0}}))
                loop.call_soon_threadsafe(q.put_nowait, _sse("message_stop", {"type": "message_stop"}))
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        _executor.submit(run)

        # Drain queue — buffer chunks until we know if this is a retry situation
        chunks = []
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            chunks.append(chunk)

        # Check if we need to retry due to context length error
        if context_error.get('retry') and attempt < max_retries - 1:
            current_max = converse_body.get("inferenceConfig", {}).get("maxTokens", 4096)
            context_length = MODEL_CONTEXT_LENGTHS.get(model_alias, 200000)
            if 'actual_input' in context_error:
                new_max = max(512, context_length - context_error['actual_input'] - 2)
            elif 'max_input' in context_error:
                new_max = max(512, context_length - context_error['max_input'] - 2)
            else:
                new_max = max(512, current_max // 2)
            print(f"[SSE] Context overflow retry {attempt + 1}: reducing maxTokens {current_max} -> {new_max}")
            if "inferenceConfig" not in converse_body:
                converse_body["inferenceConfig"] = {}
            converse_body["inferenceConfig"]["maxTokens"] = new_max
            continue  # Retry with new max_tokens

        # No retry needed — yield all buffered chunks
        for chunk in chunks:
            yield chunk
        return


@app.post("/v1/messages")
async def messages(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    model_alias = body.get("model", DEFAULT_MODEL)
    stream = body.get("stream", False)

    print(f"[MESSAGES API] model={model_alias}, stream={stream}, msg_count={len(body.get('messages', []))}")
    # Log first and last message for debugging
    msgs = body.get("messages", [])
    if msgs:
        first = msgs[0]
        last = msgs[-1]
        print(f"[MESSAGES API] first msg role={first.get('role')}, last msg role={last.get('role')}")
        # Preview content size
        first_content = first.get("content", "")[:100] if isinstance(first.get("content"), str) else str(first.get("content", ""))[:100]
        last_content = last.get("content", "")[:100] if isinstance(last.get("content"), str) else str(last.get("content", ""))[:100]
        print(f"[MESSAGES API] first content: {first_content}...")
        print(f"[MESSAGES API] last content: {last_content}...")

    try:
        bedrock_model_id, converse_body = anthropic_to_converse(body)
    except Exception as e:
        print(f"[MESSAGES API] conversion error: {e}")
        raise HTTPException(status_code=400, detail=f"Request conversion error: {e}")

    # Get client with model-specific timeout if needed
    client = get_bedrock_client(model_alias)

    if stream:
        print(f"[MESSAGES API] Using SSE streaming for {model_alias}")
        # Debug: log full request URL for tracking
        print(f"[MESSAGES API] Request URL: {request.url}")
        # Debug: log system prompt presence
        system = body.get("system")
        if system:
            system_preview = str(system)[:200] if isinstance(system, str) else str(system)[:200]
            print(f"[MESSAGES API] system prompt: {system_preview}...")
        # Debug: log tool definitions
        tools = body.get("tools", [])
        print(f"[MESSAGES API] tools count: {len(tools)}")
        return StreamingResponse(
            _sse_generator(client, bedrock_model_id, converse_body, model_alias),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        print(f"[MESSAGES API] Calling Bedrock: {bedrock_model_id}")
        resp = await asyncio.get_event_loop().run_in_executor(
            _executor, lambda: _call_converse_with_retry(client, bedrock_model_id, converse_body, model_alias)
        )
        print(f"[MESSAGES API] Bedrock call succeeded, output tokens: {resp.get('usage', {}).get('outputTokens', 0)}")
    except Exception as e:
        err_msg = str(e)
        timeout_info = f" (timeout: {MODEL_READ_TIMEOUTS.get(model_alias, 300)}s)" if "timeout" in err_msg.lower() or "read timeout" in err_msg.lower() else ""
        print(f"bedrock proxy [ERROR] Bedrock call failed: {err_msg}{timeout_info}")
        print(f"bedrock proxy [ERROR] Model: {model_alias} -> {bedrock_model_id}")
        return JSONResponse(
            status_code=500,
            content={"type": "error", "error": {"type": "api_error", "message": err_msg}},
        )

    anthropic_resp = converse_to_anthropic(model_alias, resp)
    return JSONResponse(content=anthropic_resp)


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Rough token estimation: ~4 chars per token
    total_chars = 0
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total_chars += len(str(block.get("text", "") or block.get("input", "")))
    system = body.get("system", "")
    if isinstance(system, str):
        total_chars += len(system)
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict):
                total_chars += len(block.get("text", ""))

    input_tokens = max(1, total_chars // 4)
    return JSONResponse(content={"input_tokens": input_tokens})


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/mcp/readonly")
async def mcp_readonly(request: Request):
    """Minimal MCP JSON-RPC 2.0 server — lets Copilot CLI initialize its tool system."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    rpc_id = body.get("id")
    method = body.get("method", "")

    if method == "initialize":
        result = {
            "protocolVersion": body.get("params", {}).get("protocolVersion", "2025-11-25"),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "bedrock-proxy-mcp", "version": "1.0.0"},
        }
    elif method == "tools/list":
        result = {"tools": []}
    elif method == "notifications/initialized":
        # Notification — no response needed
        return JSONResponse(content={})
    else:
        result = {}

    return JSONResponse(content={"jsonrpc": "2.0", "id": rpc_id, "result": result})


@app.get("/copilot_internal/user")
async def copilot_user(request: Request):
    """Fake GitHub Copilot user endpoint — returns unlimited quota so copilot bypasses GitHub auth."""
    # Derive base URL from the incoming request so port overrides work
    base_url = f"{request.url.scheme}://{request.url.netloc}"
    return JSONResponse(content={
        "login": "user",
        "chat_enabled": True,
        "copilot_plan": "business",
        "access_type_sku": "business",
        "codex_agent_enabled": True,
        "is_mcp_enabled": True,
        "restricted_telemetry": False,
        "unlimited": True,
        "quota_remaining": 999999,
        "percent_remaining": 100,
        "endpoints": {
            "api": base_url,
            "proxy": base_url,
            "telemetry": base_url,
            "origin-tracker": base_url,
        },
    })


@app.post("/telemetry")
@app.post("/v1/telemetry")
async def telemetry_sink(request: Request):
    """Accept and discard telemetry data."""
    return JSONResponse(content={"status": "ok"})


@app.post("/v1/engines/{engine_id}/completions")
async def completions_legacy(engine_id: str, request: Request):
    """Legacy completions endpoint — redirect to chat completions."""
    return JSONResponse(content={"choices": [{"text": "", "finish_reason": "stop"}]})


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": k, "object": "model", "owned_by": "bedrock"}
            for k in MODEL_MAP
        ],
    }


@app.get("/v1/models/{model_id:path}")
async def get_model(model_id: str):
    if model_id in MODEL_MAP:
        return {"id": model_id, "object": "model", "owned_by": "bedrock"}
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


# Copilot CLI routes without /v1 prefix (uses COPILOT_API_URL which has no /v1)
@app.get("/models")
async def list_models_copilot(request: Request):
    # Return copilotUrl to tell Copilot where to send model requests
    base_url = str(request.base_url).rstrip('/')
    models_list = [
        {"id": k, "object": "model", "owned_by": "bedrock", "model_picker_enabled": True}
        for k in MODEL_MAP
    ]
    return {
        "object": "list",
        "data": models_list,
        "models": models_list,  # SDK might expect this instead of data
        "copilotUrl": base_url  # Tell Copilot to send model requests here
    }


@app.post("/chat/completions")
async def chat_completions_copilot(request: Request):
    # Debug: log request details
    body = await request.body()
    print(f"[CHAT COMPLETIONS] Received request, {len(body)} bytes")
    print(f"[CHAT COMPLETIONS] Headers: {dict(request.headers)}")
    print(f"[CHAT COMPLETIONS] URL: {request.url}")
    return await chat_completions(request)


@app.post("/responses")
async def responses_copilot(request: Request):
    return await responses_endpoint(request)


def openai_to_anthropic(body: dict) -> dict:
    """Convert OpenAI chat completions request to Anthropic messages format."""
    messages = body.get("messages", [])
    system_text = None
    anthropic_messages = []

    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role")

        if role == "system":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
            system_text = content
            i += 1

        elif role == "tool":
            # Collect consecutive tool messages into one user turn
            tool_results = []
            while i < len(messages) and messages[i].get("role") == "tool":
                m = messages[i]
                result_content = m.get("content", "")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id", ""),
                    "content": result_content if isinstance(result_content, str) else str(result_content),
                })
                i += 1
            anthropic_messages.append({"role": "user", "content": tool_results})

        elif role == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                anthropic_content = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        anthropic_content.append({"type": "text", "text": block.get("text", "")})
                    elif block.get("type") == "image_url":
                        url = block.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            media_type, b64 = url[5:].split(",", 1)
                            media_type = media_type.split(";")[0]
                            anthropic_content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}})
                        else:
                            anthropic_content.append({"type": "image", "source": {"type": "url", "url": url}})
            else:
                anthropic_content = content
            anthropic_messages.append({"role": "user", "content": anthropic_content})
            i += 1

        elif role == "assistant":
            content = msg.get("content", "") or ""
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                anthropic_content = []
                if content:
                    anthropic_content.append({"type": "text", "text": content})
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    try:
                        fn_input = json.loads(fn.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        fn_input = {}
                    anthropic_content.append({
                        "type": "tool_use",
                        "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                        "name": fn.get("name", ""),
                        "input": fn_input,
                    })
            else:
                anthropic_content = content
            anthropic_messages.append({"role": "assistant", "content": anthropic_content})
            i += 1

        else:
            i += 1

    anthropic_body = {
        "model": body.get("model", DEFAULT_MODEL),
        "messages": anthropic_messages,
        "max_tokens": body.get("max_tokens", 8192),
        "stream": body.get("stream", False),
    }
    if system_text:
        anthropic_body["system"] = system_text
    if "temperature" in body:
        anthropic_body["temperature"] = body["temperature"]
    if "top_p" in body:
        anthropic_body["top_p"] = body["top_p"]
    if "stop" in body:
        stop = body["stop"]
        anthropic_body["stop_sequences"] = [stop] if isinstance(stop, str) else stop

    tools = body.get("tools", [])
    if tools:
        anthropic_tools = []
        for t in tools:
            if t.get("type") == "function":
                fn = t.get("function", {})
                if fn.get("name"):
                    anthropic_tools.append({
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                    })
            # skip built-in tools without a name
        if anthropic_tools:
            anthropic_body["tools"] = anthropic_tools

    return anthropic_body


_FINISH_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
}


def anthropic_to_openai_response(model_alias: str, anthropic_resp: dict) -> dict:
    """Convert Anthropic messages response to OpenAI chat completions format."""
    content_blocks = anthropic_resp.get("content", [])
    text_parts = []
    tool_calls = []

    for block in content_blocks:
        if block.get("type") == "text":
            text_parts.append(block["text"])
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block["id"],
                "type": "function",
                "function": {"name": block["name"], "arguments": json.dumps(block.get("input", {}))},
            })

    message = {"role": "assistant", "content": "\n".join(text_parts) if text_parts else None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    finish_reason = _FINISH_REASON_MAP.get(anthropic_resp.get("stop_reason", "end_turn"), "stop")
    usage = anthropic_resp.get("usage", {})
    prompt_tokens = usage.get("input_tokens", 0)
    completion_tokens = usage.get("output_tokens", 0)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_alias,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _stream_openai_sse(client, bedrock_model_id: str, converse_body: dict, model_alias: str):
    """Blocking generator: calls Bedrock converse_stream and yields OpenAI-format SSE strings."""
    resp = client.converse_stream(modelId=bedrock_model_id, **converse_body)
    stream = resp.get("stream", [])

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def chunk(delta, finish_reason=None):
        return f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_alias, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish_reason}]})}\n\n"

    yield chunk({"role": "assistant", "content": ""})

    tool_index = -1
    in_tool_block = False
    finish_reason = "stop"

    for event in stream:
        if "contentBlockStart" in event:
            start = event["contentBlockStart"].get("start", {})
            if "toolUse" in start:
                tu = start["toolUse"]
                tool_index += 1
                in_tool_block = True
                yield chunk({"tool_calls": [{"index": tool_index, "id": tu["toolUseId"], "type": "function", "function": {"name": tu["name"], "arguments": ""}}]})
            else:
                in_tool_block = False

        elif "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta", {})
            if "text" in delta:
                yield chunk({"content": delta["text"]})
            elif "toolUse" in delta:
                partial = delta["toolUse"].get("input", "")
                yield chunk({"tool_calls": [{"index": tool_index, "function": {"arguments": partial}}]})

        elif "contentBlockStop" in event:
            in_tool_block = False

        elif "messageStop" in event:
            br = event["messageStop"].get("stopReason", "end_turn")
            finish_reason = _FINISH_REASON_MAP.get(br, "stop")

    yield chunk({}, finish_reason=finish_reason)
    yield "data: [DONE]\n\n"


async def _openai_sse_generator(client, bedrock_model_id: str, converse_body: dict, model_alias: str):
    """Async wrapper for OpenAI-format SSE streaming."""
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def run():
        try:
            for chunk in _stream_openai_sse(client, bedrock_model_id, converse_body, model_alias):
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
        except Exception as e:
            print(f"[OPENAI SSE] ERROR: {str(e)}")
            # Send [DONE] without error to prevent client treating as streaming error
            loop.call_soon_threadsafe(queue.put_nowait, "data: [DONE]\n\n")
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    _executor.submit(run)

    while True:
        chunk = await queue.get()
        if chunk is None:
            break
        yield chunk


def responses_to_anthropic(body: dict) -> dict:
    """Convert OpenAI Responses API request to Anthropic messages format."""
    raw_input = body.get("input", [])
    if isinstance(raw_input, str):
        raw_input = [{"role": "user", "content": raw_input}]

    system_text = body.get("instructions")
    anthropic_messages = []

    i = 0
    while i < len(raw_input):
        item = raw_input[i]
        item_type = item.get("type")
        role = item.get("role")

        if item_type == "function_call":
            # Collect ALL consecutive function_call items into ONE assistant message
            tool_uses = []
            while i < len(raw_input) and raw_input[i].get("type") == "function_call":
                m = raw_input[i]
                try:
                    fn_input = json.loads(m.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    fn_input = {}
                tool_uses.append({
                    "type": "tool_use",
                    "id": m.get("call_id", f"toolu_{uuid.uuid4().hex[:24]}"),
                    "name": m.get("name", ""),
                    "input": fn_input,
                })
                i += 1
            # Merge into previous assistant message if possible, otherwise create new one
            if anthropic_messages and anthropic_messages[-1]["role"] == "assistant":
                prev = anthropic_messages[-1]
                if isinstance(prev["content"], str):
                    prev["content"] = [{"type": "text", "text": prev["content"]}] + tool_uses
                elif isinstance(prev["content"], list):
                    prev["content"] = prev["content"] + tool_uses
                else:
                    prev["content"] = tool_uses
            else:
                anthropic_messages.append({"role": "assistant", "content": tool_uses})

        elif item_type == "function_call_output":
            # Collect ALL consecutive function_call_output into ONE user message
            tool_results = []
            while i < len(raw_input) and raw_input[i].get("type") == "function_call_output":
                m = raw_input[i]
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": m.get("call_id", ""),
                    "content": m.get("output", ""),
                })
                i += 1
            anthropic_messages.append({"role": "user", "content": tool_results})

        elif role == "system":
            content = item.get("content", "")
            if isinstance(content, list):
                content = "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
            system_text = content
            i += 1

        elif role == "user":
            content = item.get("content", "")
            if isinstance(content, list):
                anthropic_content = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type", "")
                    if btype in ("input_text", "text"):
                        anthropic_content.append({"type": "text", "text": block.get("text", "")})
                    elif btype == "input_image":
                        url = block.get("image_url", "")
                        if isinstance(url, dict):
                            url = url.get("url", "")
                        if url.startswith("data:"):
                            media_type, b64 = url[5:].split(",", 1)
                            anthropic_content.append({"type": "image", "source": {"type": "base64", "media_type": media_type.split(";")[0], "data": b64}})
                        else:
                            anthropic_content.append({"type": "image", "source": {"type": "url", "url": url}})
            else:
                anthropic_content = content
            anthropic_messages.append({"role": "user", "content": anthropic_content})
            i += 1

        elif role == "assistant":
            content = item.get("content", [])
            if isinstance(content, str):
                anthropic_content = content
            elif isinstance(content, list):
                anthropic_content = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type", "")
                    if btype in ("output_text", "text"):
                        anthropic_content.append({"type": "text", "text": block.get("text", "")})
                    elif btype == "tool_use":
                        anthropic_content.append(block)
            else:
                anthropic_content = str(content) if content else ""
            anthropic_messages.append({"role": "assistant", "content": anthropic_content})
            i += 1

        else:
            i += 1

    anthropic_body = {
        "model": body.get("model", DEFAULT_MODEL),
        "messages": anthropic_messages,
        "max_tokens": body.get("max_output_tokens", 8192),
        "stream": body.get("stream", False),
    }
    if system_text:
        anthropic_body["system"] = system_text
    if "temperature" in body:
        anthropic_body["temperature"] = body["temperature"]
    if "top_p" in body:
        anthropic_body["top_p"] = body["top_p"]

    tools = body.get("tools", [])
    if tools:
        anthropic_tools = []
        for t in tools:
            if t.get("type") == "function" and t.get("name"):
                anthropic_tools.append({
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
                })
            # skip built-in OpenAI tools (web_search, file_search, etc.) — not supported on Bedrock
        if anthropic_tools:
            anthropic_body["tools"] = anthropic_tools

    return anthropic_body


def anthropic_to_responses(model_alias: str, anthropic_resp: dict) -> dict:
    """Convert Anthropic messages response to OpenAI Responses API format."""
    content_blocks = anthropic_resp.get("content", [])
    resp_id = f"resp_{uuid.uuid4().hex[:24]}"
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    msg_content = []
    output_items = []

    for block in content_blocks:
        if block.get("type") == "text":
            msg_content.append({"type": "output_text", "text": block["text"], "annotations": []})
        elif block.get("type") == "tool_use":
            output_items.append({
                "id": f"fc_{uuid.uuid4().hex[:24]}",
                "type": "function_call",
                "call_id": block["id"],
                "name": block["name"],
                "arguments": json.dumps(block.get("input", {})),
                "status": "completed",
            })

    if msg_content:
        output_items.insert(0, {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": msg_content,
        })

    usage = anthropic_resp.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    return {
        "id": resp_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model_alias,
        "output": output_items,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def _stream_responses_sse(client, bedrock_model_id: str, converse_body: dict, model_alias: str):
    """Blocking generator: yields OpenAI Responses API SSE events from Bedrock stream.

    Bedrock skips contentBlockStart for text blocks — deltas arrive directly.
    Tool blocks DO emit contentBlockStart with toolUse info before their deltas.
    """
    resp_id = f"resp_{uuid.uuid4().hex[:24]}"
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    created_at = int(time.time())

    def sse(event_type, data):
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    base_resp = {"id": resp_id, "object": "response", "created_at": created_at, "model": model_alias, "status": "in_progress", "output": []}
    yield sse("response.created", {"type": "response.created", "response": base_resp})
    yield sse("response.in_progress", {"type": "response.in_progress", "response": base_resp})

    output_index = 0
    in_tool_block = False
    msg_item_added = False
    msg_item_closed = False
    full_text = ""
    tool_items = []
    input_tokens = 0
    output_tokens = 0

    try:
        resp = client.converse_stream(modelId=bedrock_model_id, **converse_body)
        stream = resp.get("stream", [])

        for event in stream:
            if "contentBlockStart" in event:
                # Only tool blocks emit contentBlockStart; text blocks go straight to delta
                start = event["contentBlockStart"].get("start", {})
                if "toolUse" in start:
                    tu = start["toolUse"]
                    in_tool_block = True
                    fc_id = f"fc_{uuid.uuid4().hex[:24]}"
                    fc_item = {"id": fc_id, "type": "function_call", "call_id": tu["toolUseId"], "name": tu["name"], "arguments": "", "status": "in_progress"}
                    tool_items.append(fc_item)
                    yield sse("response.output_item.added", {"type": "response.output_item.added", "output_index": output_index, "item": fc_item})

            elif "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                if "text" in delta:
                    text = delta["text"]
                    full_text += text
                    # Lazily open the message output item on first text delta
                    if not msg_item_added:
                        msg_item_added = True
                        yield sse("response.output_item.added", {"type": "response.output_item.added", "output_index": output_index, "item": {"id": msg_id, "type": "message", "role": "assistant", "status": "in_progress", "content": []}})
                        yield sse("response.content_part.added", {"type": "response.content_part.added", "item_id": msg_id, "output_index": output_index, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}})
                    yield sse("response.output_text.delta", {"type": "response.output_text.delta", "item_id": msg_id, "output_index": output_index, "content_index": 0, "delta": text})
                elif "toolUse" in delta and in_tool_block and tool_items:
                    partial = delta["toolUse"].get("input", "")
                    tool_items[-1]["arguments"] += partial
                    yield sse("response.function_call_arguments.delta", {"type": "response.function_call_arguments.delta", "item_id": tool_items[-1]["id"], "output_index": output_index, "delta": partial})

            elif "contentBlockStop" in event:
                if in_tool_block and tool_items:
                    fc = tool_items[-1]
                    fc["status"] = "completed"
                    yield sse("response.function_call_arguments.done", {"type": "response.function_call_arguments.done", "item_id": fc["id"], "output_index": output_index, "arguments": fc["arguments"]})
                    yield sse("response.output_item.done", {"type": "response.output_item.done", "output_index": output_index, "item": fc})
                    in_tool_block = False
                    output_index += 1
                elif msg_item_added and not msg_item_closed:
                    msg_item_closed = True
                    yield sse("response.output_text.done", {"type": "response.output_text.done", "item_id": msg_id, "output_index": output_index, "content_index": 0, "text": full_text})
                    yield sse("response.content_part.done", {"type": "response.content_part.done", "item_id": msg_id, "output_index": output_index, "content_index": 0, "part": {"type": "output_text", "text": full_text, "annotations": []}})
                    yield sse("response.output_item.done", {"type": "response.output_item.done", "output_index": output_index, "item": {"id": msg_id, "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": full_text, "annotations": []}]}})
                    output_index += 1

            elif "metadata" in event:
                u = event["metadata"].get("usage", {})
                input_tokens = u.get("inputTokens", 0)
                output_tokens = u.get("outputTokens", 0)

    except Exception as e:
        err_msg = str(e)
        print(f"bedrock proxy [ERROR] Stream failed for {model_alias}: {err_msg}")

    final_output = []
    if msg_item_added:
        final_output.append({"id": msg_id, "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": full_text, "annotations": []}]})
    final_output.extend(tool_items)

    final_resp = {
        "id": resp_id, "object": "response", "created_at": created_at, "status": "completed", "model": model_alias,
        "output": final_output,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens, "output_tokens_details": {"reasoning_tokens": 0}},
    }
    yield sse("response.completed", {"type": "response.completed", "response": final_resp})


async def _responses_sse_generator(client, bedrock_model_id: str, converse_body: dict, model_alias: str):
    """Async wrapper for Responses API SSE streaming."""
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def run():
        try:
            for chunk in _stream_responses_sse(client, bedrock_model_id, converse_body, model_alias):
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
        except Exception as e:
            print(f"[RESPONSES SSE] ERROR: {str(e)}")
            # Don't emit error event - let stream end naturally to prevent client treating as streaming error
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    _executor.submit(run)

    while True:
        chunk = await queue.get()
        if chunk is None:
            break
        yield chunk


@app.post("/v1/responses")
async def responses_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    anthropic_body = responses_to_anthropic(body)
    model_alias = anthropic_body.get("model", DEFAULT_MODEL)
    stream = anthropic_body.get("stream", False)

    try:
        bedrock_model_id, converse_body = anthropic_to_converse(anthropic_body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Request conversion error: {e}")

    client = get_bedrock_client(model_alias)

    if stream:
        return StreamingResponse(
            _responses_sse_generator(client, bedrock_model_id, converse_body, model_alias),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        resp = await asyncio.get_event_loop().run_in_executor(
            _executor, lambda: _call_converse_with_retry(client, bedrock_model_id, converse_body, model_alias)
        )
    except Exception as e:
        err_msg = str(e)
        print(f"bedrock proxy [ERROR] Bedrock call failed: {err_msg}")
        return JSONResponse(status_code=500, content={"error": {"message": err_msg, "type": "api_error"}})

    anthropic_resp = converse_to_anthropic(model_alias, resp)
    return JSONResponse(content=anthropic_to_responses(model_alias, anthropic_resp))


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    anthropic_body = openai_to_anthropic(body)
    model_alias = anthropic_body.get("model", DEFAULT_MODEL)
    stream = anthropic_body.get("stream", False)

    try:
        bedrock_model_id, converse_body = anthropic_to_converse(anthropic_body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Request conversion error: {e}")

    client = get_bedrock_client(model_alias)

    if stream:
        return StreamingResponse(
            _openai_sse_generator(client, bedrock_model_id, converse_body, model_alias),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        resp = await asyncio.get_event_loop().run_in_executor(
            _executor, lambda: _call_converse_with_retry(client, bedrock_model_id, converse_body, model_alias)
        )
    except Exception as e:
        err_msg = str(e)
        print(f"bedrock proxy [ERROR] Bedrock call failed: {err_msg}")
        print(f"bedrock proxy [ERROR] Model: {model_alias} -> {bedrock_model_id}")
        return JSONResponse(status_code=500, content={"error": {"message": err_msg, "type": "api_error"}})

    anthropic_resp = converse_to_anthropic(model_alias, resp)
    return JSONResponse(content=anthropic_to_openai_response(model_alias, anthropic_resp))


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def catch_all(path: str, request: Request):
    body = await request.body()
    print(f"[CATCH-ALL] {request.method} /{path}")
    print(f"[CATCH-ALL] Headers: {dict(request.headers)}")
    if body:
        try:
            print(f"[CATCH-ALL] Body: {json.loads(body)}")
        except Exception:
            print(f"[CATCH-ALL] Body (raw): {body[:200]}")
    # Return empty success for telemetry/tracking endpoints to avoid errors
    if any(kw in path for kw in ("telemetry", "tracking", "usage", "log")):
        return JSONResponse(content={"status": "ok"})
    return JSONResponse(status_code=404, content={
        "type": "error",
        "error": {"type": "not_found_error", "message": f"Endpoint /{path} not found"}
    })


if __name__ == "__main__":
    port = int(os.environ.get("PROXY_PORT", 4002))
    print(f"Starting Bedrock DeepSeek proxy on port {port}")
    print(f"AWS Region: {AWS_REGION}")
    print(f"Models: {list(MODEL_MAP.keys())}")
    print(f"Default timeout: 300s (5 minutes)")
    if MODEL_READ_TIMEOUTS:
        print(f"Custom timeouts configured for: {list(MODEL_READ_TIMEOUTS.keys())}")
    print(f"Retry policy: 3 attempts with adaptive exponential backoff")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        timeout_keep_alive=0,  # Disable timeout for long-running AI requests
        timeout_graceful_shutdown=30  # 30 seconds for graceful shutdown
    )
