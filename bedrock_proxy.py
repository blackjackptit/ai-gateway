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
import os
import boto3
from botocore.config import Config
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()
_executor = ThreadPoolExecutor()

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
    "claude-sonnet-4-5":      "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-sonnet-4-6":      "us.anthropic.claude-sonnet-4-6",
    "claude-opus-4-5":        "us.anthropic.claude-opus-4-5-20251101-v1:0",
    "claude-opus-4-6":        "us.anthropic.claude-opus-4-6-v1",
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
    "claude-haiku-4-5", "claude-sonnet-4-5", "claude-sonnet-4-6",
    "claude-opus-4-5", "claude-opus-4-6", "claude-opus-4", "claude-sonnet-4",
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
    # MiniMax models (1M context, cap output to prevent timeouts)
    "minimax-m2":   8192,
    "minimax-m2.1": 8192,
    "minimax-m2.5": 8192,
    # GLM models (128K context, conservative limit)
    "glm-5":        8192,
    "glm-5-plus":   8192,
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
            converse_content = [{"text": content}]
        else:
            # Handle content blocks
            converse_content = []
            for block in content:
                if isinstance(block, str):
                    converse_content.append({"text": block})
                elif isinstance(block, dict):
                    btype = block.get("type")
                    if btype == "text":
                        converse_content.append({"text": block["text"]})
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
                                tc = [{"text": result_content}]
                            elif isinstance(result_content, list):
                                tc = [
                                    {"text": b["text"]}
                                    for b in result_content
                                    if isinstance(b, dict) and b.get("type") == "text"
                                ]
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
                                "text": f"[Tool result: {result_content}]"
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
    if "top_p" in body:
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


def _stream_bedrock_sse(client, bedrock_model_id: str, converse_body: dict, model_alias: str):
    """Blocking generator that calls converse_stream and yields SSE strings."""
    resp = client.converse_stream(modelId=bedrock_model_id, **converse_body)
    stream = resp.get("stream", [])

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

    block_index = 0
    stop_reason = "end_turn"
    output_tokens = 0

    for event in stream:
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

        elif "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta", {})
            if "text" in delta:
                yield _sse("content_block_delta", {
                    "type": "content_block_delta", "index": block_index,
                    "delta": {"type": "text_delta", "text": delta["text"]},
                })
            elif "toolUse" in delta:
                yield _sse("content_block_delta", {
                    "type": "content_block_delta", "index": block_index,
                    "delta": {"type": "input_json_delta", "partial_json": delta["toolUse"].get("input", "")},
                })

        elif "contentBlockStop" in event:
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
            block_index += 1

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
    """Async wrapper: runs the blocking SSE generator in a thread and yields chunks."""
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def run():
        try:
            for chunk in _stream_bedrock_sse(client, bedrock_model_id, converse_body, model_alias):
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
        except Exception as e:
            err_msg = str(e)
            timeout_info = f" (timeout: {MODEL_READ_TIMEOUTS.get(model_alias, 300)}s)" if "timeout" in err_msg.lower() or "read timeout" in err_msg.lower() else ""
            print(f"bedrock proxy [ERROR] Stream failed: {err_msg}{timeout_info}")
            print(f"bedrock proxy [ERROR] Model: {model_alias} -> {bedrock_model_id}")
            err = _sse("error", {"type": "error", "error": {"type": "api_error", "message": err_msg}})
            loop.call_soon_threadsafe(queue.put_nowait, err)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    _executor.submit(run)

    while True:
        chunk = await queue.get()
        if chunk is None:
            break
        yield chunk


@app.post("/v1/messages")
async def messages(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    model_alias = body.get("model", DEFAULT_MODEL)
    stream = body.get("stream", False)

    try:
        bedrock_model_id, converse_body = anthropic_to_converse(body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Request conversion error: {e}")

    # Get client with model-specific timeout if needed
    client = get_bedrock_client(model_alias)

    if stream:
        return StreamingResponse(
            _sse_generator(client, bedrock_model_id, converse_body, model_alias),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        resp = await asyncio.get_event_loop().run_in_executor(
            _executor, lambda: client.converse(modelId=bedrock_model_id, **converse_body)
        )
    except Exception as e:
        err_msg = str(e)
        timeout_info = f" (timeout: {MODEL_READ_TIMEOUTS.get(model_alias, 300)}s)" if "timeout" in err_msg.lower() or "read timeout" in err_msg.lower() else ""
        print(f"bedrock proxy [ERROR] Bedrock call failed: {err_msg}{timeout_info}")
        print(f"bedrock proxy [ERROR] Model: {model_alias} -> {bedrock_model_id}")
        return JSONResponse(
            status_code=500,
            content={"error": {"type": "api_error", "message": err_msg}},
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


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": k, "object": "model", "owned_by": "bedrock"}
            for k in MODEL_MAP
        ],
    }


if __name__ == "__main__":
    port = int(os.environ.get("PROXY_PORT", 4002))
    print(f"Starting Bedrock DeepSeek proxy on port {port}")
    print(f"AWS Region: {AWS_REGION}")
    print(f"Models: {list(MODEL_MAP.keys())}")
    print(f"Default timeout: 300s (5 minutes)")
    if MODEL_READ_TIMEOUTS:
        print(f"Custom timeouts configured for: {list(MODEL_READ_TIMEOUTS.keys())}")
    print(f"Retry policy: 3 attempts with adaptive exponential backoff")
    uvicorn.run(app, host="0.0.0.0", port=port)
