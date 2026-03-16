"""
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

import json
import uuid
import time
import os
import boto3
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()

# Model ID mapping
MODEL_MAP = {
    # DeepSeek
    "deepseek-r1":   "us.deepseek.r1-v1:0",
    "deepseek-v3":   "deepseek.v3.2",
    "deepseek-v3.2": "deepseek.v3.2",
    # Qwen3 (32K context)
    "qwen3-32b":         "qwen.qwen3-32b-v1:0",
    "qwen3-coder":       "qwen.qwen3-coder-30b-a3b-v1:0",
    "qwen3-coder-next":  "qwen.qwen3-coder-next",
    "qwen3-80b":         "qwen.qwen3-next-80b-a3b",
    "qwen3-vl":          "qwen.qwen3-vl-235b-a22b",
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
    "minimax-m2", "minimax-m2.1",
    # Mistral
    "mistral-large-3", "magistral-small", "devstral-2", "pixtral-large",
    # Kimi — outputs raw text markup, NOT added here (not supported)
    # "kimi-k2", "kimi-k2-thinking",
}

# Hard cap on output tokens for small-context models (32K window)
MODEL_MAX_TOKENS = {
    "qwen3-32b":        4000,
    "qwen3-coder":      4000,
    "qwen3-coder-next": 4000,
}

# Max system prompt characters for small-context models
MODEL_MAX_SYSTEM_CHARS = {
    "qwen3-32b":        8000,
    "qwen3-coder":      8000,
    "qwen3-coder-next": 8000,
}

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")


def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name=AWS_REGION)


DEFAULT_MODEL = "deepseek-r1"

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


@app.post("/v1/messages")
async def messages(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    model_alias = body.get("model", "deepseek-r1")

    try:
        bedrock_model_id, converse_body = anthropic_to_converse(body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Request conversion error: {e}")

    try:
        client = get_bedrock_client()
        resp = client.converse(modelId=bedrock_model_id, **converse_body)
    except Exception as e:
        err_msg = str(e)
        print(f"[ERROR] Bedrock call failed: {err_msg}")
        return JSONResponse(
            status_code=500,
            content={"error": {"type": "api_error", "message": err_msg}},
        )

    anthropic_resp = converse_to_anthropic(model_alias, resp)
    return JSONResponse(content=anthropic_resp)


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
    uvicorn.run(app, host="0.0.0.0", port=port)
