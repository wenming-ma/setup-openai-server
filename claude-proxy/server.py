from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import httpx
import time
import uuid
import json
import os
import re
import base64
import logging
from datetime import datetime
import concurrent.futures
import asyncio
import requests

# 配置日志
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 错误日志文件
error_handler = logging.FileHandler(os.path.join(LOG_DIR, "error.log"), encoding="utf-8")
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(error_handler)

# 调试日志文件 - 记录原始响应用于排查问题
debug_handler = logging.FileHandler(os.path.join(LOG_DIR, "debug.log"), encoding="utf-8")
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(debug_handler)

app = FastAPI(title="OpenAI to Claude Proxy Server")

# Claude API 配置 (可动态更新，支持持久化)
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

class Config:
    def __init__(self):
        # 默认值
        self.base_url = "http://115.175.23.49:3000/api"
        self.api_key = "cr_b11e7fecd0961b3503a7a7019159d75513aea6c199f9352780c171dfa1b1d54d"
        # 尝试从文件加载
        self.load()

    def load(self):
        """从文件加载配置"""
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.base_url = data.get("base_url", self.base_url)
                    self.api_key = data.get("api_key", self.api_key)
                    logger.info(f"Loaded config from {CONFIG_FILE}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")

    def save(self):
        """保存配置到文件"""
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "base_url": self.base_url,
                    "api_key": self.api_key
                }, f, indent=2)
            logger.info(f"Saved config to {CONFIG_FILE}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

config = Config()

# 保存用户消息的文件夹
MESSAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_messages")
os.makedirs(MESSAGES_DIR, exist_ok=True)


# stop_reason 映射
STOP_REASON_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls"
}

# 模型映射 - 服务器启动时动态获取并验证
# tiny-model -> 最新可用的 sonnet, bigger-model -> 最新可用的 opus
MODEL_MAP = {}

def test_model(model_name: str) -> bool:
    """测试模型是否真正可用"""
    try:
        response = requests.post(
            f"{config.base_url}/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": config.api_key,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": model_name,
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}]
            },
            timeout=30
        )
        if response.status_code == 200:
            return True
        logger.warning(f"Model {model_name} test failed: {response.status_code} - {response.text[:100]}")
        return False
    except Exception as e:
        logger.warning(f"Model {model_name} test error: {e}")
        return False

def init_model_map():
    """从上游API获取模型列表，动态设置映射"""
    global MODEL_MAP

    try:
        response = requests.get(
            f"{config.base_url}/v1/models",
            headers={
                "x-api-key": config.api_key,
                "anthropic-version": "2023-06-01"
            },
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            models = [m["id"] for m in data.get("data", [])]

            # 找 sonnet 模型列表，按名称倒序（新版本优先）
            sonnet_models = sorted([m for m in models if "sonnet" in m.lower()], reverse=True)
            # 找 opus 模型列表
            opus_models = sorted([m for m in models if "opus" in m.lower()], reverse=True)

            # 测试并选择第一个可用的 sonnet
            latest_sonnet = None
            for model in sonnet_models:
                logger.info(f"Testing sonnet model: {model}")
                if test_model(model):
                    latest_sonnet = model
                    logger.info(f"Found working sonnet model: {model}")
                    break

            # 测试并选择第一个可用的 opus
            latest_opus = None
            for model in opus_models:
                logger.info(f"Testing opus model: {model}")
                if test_model(model):
                    latest_opus = model
                    logger.info(f"Found working opus model: {model}")
                    break

            # 如果没找到可用模型，使用默认值
            if not latest_sonnet:
                latest_sonnet = "claude-sonnet-4-20250514"
                logger.warning(f"No working sonnet found, using default: {latest_sonnet}")
            if not latest_opus:
                latest_opus = "claude-opus-4-20250514"
                logger.warning(f"No working opus found, using default: {latest_opus}")

            MODEL_MAP = {
                "tiny-model": latest_sonnet,
                "bigger-model": latest_opus,
                "gpt-4": latest_sonnet,
                "gpt-4o": latest_sonnet,
                "gpt-3.5-turbo": latest_sonnet,
            }

            logger.info(f"Model mapping initialized: tiny-model -> {latest_sonnet}, bigger-model -> {latest_opus}")
        else:
            logger.error(f"Failed to get models: {response.status_code}")
            MODEL_MAP = {
                "tiny-model": "claude-sonnet-4-20250514",
                "bigger-model": "claude-opus-4-20250514",
            }
    except Exception as e:
        logger.error(f"Error initializing model map: {e}")
        MODEL_MAP = {
            "tiny-model": "claude-sonnet-4-20250514",
            "bigger-model": "claude-opus-4-20250514",
        }

# 服务器启动时初始化模型映射
init_model_map()


# ==================== 用户消息保存函数 ====================

def extract_user_query(content):
    """提取纯净的用户内容，去除 <user_query> 等标签"""
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text", ""))
        content = "\n".join(texts)

    if isinstance(content, str):
        match = re.search(r'<user_query>\s*(.*?)\s*</user_query>', content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return content.strip()

    return str(content)


def _save_user_message_sync(messages: list):
    """同步保存用户消息"""
    user_content = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_content = extract_user_query(msg.get("content", ""))
            break

    if user_content:
        filepath = os.path.join(MESSAGES_DIR, "latest_message.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(user_content)


_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

def save_user_message(messages: list):
    """异步保存用户消息（不阻塞主线程）"""
    _executor.submit(_save_user_message_sync, messages)


# ==================== 工具转换函数 ====================

def convert_openai_tools_to_claude(tools: list) -> list:
    """将 OpenAI tools 格式转换为 Claude tools 格式"""
    if not tools:
        return []

    claude_tools = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool.get("function", {})
            claude_tools.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}})
            })
    return claude_tools


def convert_openai_tool_choice_to_claude(tool_choice):
    """将 OpenAI tool_choice 转换为 Claude tool_choice"""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice == "auto":
            return {"type": "auto"}
        elif tool_choice == "none":
            return None
        elif tool_choice == "required":
            return {"type": "any"}
    elif isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            func_name = tool_choice.get("function", {}).get("name", "")
            return {"type": "tool", "name": func_name}
    return {"type": "auto"}


def convert_claude_tool_use_to_openai(content_blocks: list) -> list:
    """将 Claude tool_use 转换为 OpenAI tool_calls 格式"""
    tool_calls = []
    for idx, block in enumerate(content_blocks):
        if block.get("type") == "tool_use":
            tool_calls.append({
                "id": f"call_{block.get('id', uuid.uuid4().hex[:12])}",
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {}))
                },
                "index": idx
            })
    return tool_calls


# ==================== 图片内容转换函数 ====================

async def convert_image_content(content_part: dict) -> dict:
    """将 OpenAI image_url 格式转换为 Claude image 格式"""
    image_url = content_part.get("image_url", {})
    url = image_url.get("url", "")

    if url.startswith("data:"):
        match = re.match(r'data:([^;]+);base64,(.+)', url)
        if match:
            media_type = match.group(1)
            data = match.group(2)
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data
                }
            }
    elif url.startswith("http"):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    content_type = response.headers.get("content-type", "image/jpeg")
                    data = base64.b64encode(response.content).decode("utf-8")
                    return {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": content_type.split(";")[0],
                            "data": data
                        }
                    }
        except Exception as e:
            logger.error(f"Failed to download image: {e}")

    return {"type": "text", "text": f"[Image: {url}]"}


# ==================== 消息转换函数 ====================

async def convert_openai_to_claude(openai_messages: list) -> tuple:
    """将 OpenAI 消息格式转换为 Claude 格式"""
    system_prompt = None
    claude_messages = []

    for msg in openai_messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "system":
            if isinstance(content, str):
                system_prompt = content
            elif isinstance(content, list):
                texts = [p.get("text", "") for p in content if p.get("type") == "text"]
                system_prompt = "\n".join(texts)

        elif role == "user":
            if isinstance(content, str):
                claude_messages.append({"role": "user", "content": content})
            elif isinstance(content, list):
                claude_content = []
                for part in content:
                    if part.get("type") == "text":
                        claude_content.append({"type": "text", "text": part.get("text", "")})
                    elif part.get("type") == "image_url":
                        image_block = await convert_image_content(part)
                        claude_content.append(image_block)
                claude_messages.append({"role": "user", "content": claude_content})

        elif role == "assistant":
            assistant_content = []

            # 处理文本内容
            if isinstance(content, str) and content:
                assistant_content.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        assistant_content.append({"type": "text", "text": part.get("text", "")})

            # 处理 tool_calls (OpenAI 格式)
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", {})
                    tool_id = tc.get("id", "").replace("call_", "")
                    if not tool_id:
                        tool_id = uuid.uuid4().hex[:12]
                    assistant_content.append({
                        "type": "tool_use",
                        "id": tool_id,
                        "name": func.get("name", ""),
                        "input": json.loads(func.get("arguments", "{}"))
                    })

            if assistant_content:
                claude_messages.append({"role": "assistant", "content": assistant_content})
            elif content:  # 字符串内容但不为空
                claude_messages.append({"role": "assistant", "content": content})

        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "").replace("call_", "")
            claude_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": content if isinstance(content, str) else json.dumps(content)
                }]
            })

    return system_prompt, claude_messages


# ==================== 响应转换函数 ====================

def convert_claude_to_openai(claude_response: dict, model: str) -> dict:
    """将 Claude 响应转换为 OpenAI 格式"""
    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    content_blocks = claude_response.get("content", [])
    text_content = ""
    tool_calls = []

    for idx, block in enumerate(content_blocks):
        if block.get("type") == "text":
            text_content += block.get("text", "")
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": f"call_{block.get('id', uuid.uuid4().hex[:12])}",
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {}))
                },
                "index": idx
            })

    # 构建消息
    message = {"role": "assistant", "content": text_content if text_content else None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    # 映射 stop_reason
    stop_reason = claude_response.get("stop_reason", "end_turn")
    finish_reason = STOP_REASON_MAP.get(stop_reason, "stop")

    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": claude_response.get("usage", {}).get("input_tokens", 0),
            "completion_tokens": claude_response.get("usage", {}).get("output_tokens", 0),
            "total_tokens": (
                claude_response.get("usage", {}).get("input_tokens", 0) +
                claude_response.get("usage", {}).get("output_tokens", 0)
            )
        }
    }


# ==================== 流式响应处理 ====================

class StreamingResponseIterator:
    """处理 Claude 流式响应并转换为 OpenAI 格式"""

    def __init__(self, model: str):
        self.response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        self.created = int(time.time())
        self.model = model
        self.current_tool_index = -1
        self.tool_calls_started = {}
        self.current_content_block_type = None
        self.accumulated_text = ""

    def create_chunk(self, delta: dict, finish_reason: str = None) -> str:
        """创建 OpenAI 格式的流式 chunk"""
        chunk = {
            "id": self.response_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }]
        }
        return f"data: {json.dumps(chunk)}\n\n"

    def process_event(self, event: dict) -> str:
        """处理单个 Claude 事件并返回 OpenAI 格式的 chunk"""
        event_type = event.get("type")
        result = ""

        if event_type == "message_start":
            # 发送初始 role chunk
            result = self.create_chunk({"role": "assistant", "content": ""})

        elif event_type == "content_block_start":
            content_block = event.get("content_block", {})
            self.current_content_block_type = content_block.get("type")

            if self.current_content_block_type == "tool_use":
                self.current_tool_index += 1
                tool_id = content_block.get("id", "")
                tool_name = content_block.get("name", "")

                result = self.create_chunk({
                    "tool_calls": [{
                        "index": self.current_tool_index,
                        "id": f"call_{tool_id}",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": ""
                        }
                    }]
                })

        elif event_type == "content_block_delta":
            delta = event.get("delta", {})
            delta_type = delta.get("type")

            if delta_type == "text_delta":
                text = delta.get("text", "")
                self.accumulated_text += text
                result = self.create_chunk({"content": text})

            elif delta_type == "input_json_delta":
                if self.current_content_block_type == "tool_use":
                    partial_json = delta.get("partial_json", "")
                    result = self.create_chunk({
                        "tool_calls": [{
                            "index": self.current_tool_index,
                            "function": {
                                "arguments": partial_json
                            }
                        }]
                    })

        elif event_type == "content_block_stop":
            self.current_content_block_type = None

        elif event_type == "message_delta":
            delta = event.get("delta", {})
            stop_reason = delta.get("stop_reason", "end_turn")
            finish_reason = STOP_REASON_MAP.get(stop_reason, "stop")
            result = self.create_chunk({}, finish_reason)

        elif event_type == "message_stop":
            result = "data: [DONE]\n\n"

        elif event_type == "error":
            error_msg = event.get("error", {}).get("message", "Unknown error")
            result = self.create_chunk({"content": f"Error: {error_msg}"}, "stop")
            result += "data: [DONE]\n\n"

        return result


async def stream_claude_response(claude_url: str, headers: dict, payload: dict, model: str, max_retries: int = 3):
    """流式转发 Claude 响应并转换为 OpenAI 格式，带重试机制"""
    last_error = None

    for attempt in range(max_retries):
        iterator = StreamingResponseIterator(model)

        if attempt > 0:
            wait_time = 2 ** attempt  # 指数退避: 2, 4, 8 秒
            logger.info(f"Retry attempt {attempt + 1}/{max_retries} after {wait_time}s wait")
            await asyncio.sleep(wait_time)

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream("POST", claude_url, headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        error_msg = error_text.decode()
                        logger.error(f"Claude API error (attempt {attempt + 1}): {response.status_code} - {error_msg}")

                        # 检查是否为可重试的错误
                        if response.status_code in (500, 502, 503, 504) and "ReadableStream" in error_msg:
                            last_error = error_msg
                            continue  # 重试
                        elif response.status_code == 502 and "unavailable" in error_msg.lower():
                            last_error = error_msg
                            continue  # 模型不可用，重试
                        else:
                            # 不可重试的错误，直接返回
                            yield iterator.create_chunk({"content": f"Error: {error_msg}"}, "stop")
                            yield "data: [DONE]\n\n"
                            return

                    # 成功连接，开始处理流
                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        logger.debug(f"Raw line: {line}")

                        if line.startswith("event:"):
                            continue

                        if not line.startswith("data:"):
                            continue

                        data = line[5:].strip()
                        if not data or data == "[DONE]":
                            continue

                        try:
                            event = json.loads(data)

                            # 检查是否为错误事件
                            if event.get("type") == "error" or "error" in event:
                                error_msg = event.get("error", {})
                                if isinstance(error_msg, dict):
                                    error_msg = error_msg.get("message", str(error_msg))
                                logger.error(f"Stream error event: {error_msg}")
                                last_error = str(error_msg)
                                break  # 跳出内层循环，尝试重试

                            chunk = iterator.process_event(event)
                            if chunk:
                                yield chunk

                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error: {e}, data: {data}")
                            continue

                    # 流正常完成
                    return

        except httpx.RequestError as e:
            logger.error(f"Request error (attempt {attempt + 1}): {str(e)}")
            last_error = str(e)
            continue  # 重试

    # 所有重试都失败了
    logger.error(f"All {max_retries} attempts failed. Last error: {last_error}")
    final_iterator = StreamingResponseIterator(model)
    yield final_iterator.create_chunk({"content": f"Error after {max_retries} retries: {last_error}"}, "stop")
    yield "data: [DONE]\n\n"


# ==================== API 端点 ====================

@app.get("/")
async def root():
    return {"status": "ok", "message": "OpenAI to Claude Proxy Server"}


@app.get("/test", response_class=HTMLResponse)
async def test_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>OpenAI to Claude Proxy</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 900px; margin: 50px auto; padding: 20px; }
            .status { color: green; font-size: 24px; }
            .info { background: #f0f0f0; padding: 15px; border-radius: 8px; margin: 20px 0; }
            code { background: #e0e0e0; padding: 2px 6px; border-radius: 4px; }
            .config-section { background: #e8f4f8; padding: 20px; border-radius: 8px; margin: 20px 0; border: 2px solid #4a90a4; }
            .form-group { margin: 15px 0; }
            .form-group label { display: block; font-weight: bold; margin-bottom: 5px; }
            .form-group input { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 14px; box-sizing: border-box; }
            .btn { padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; margin-right: 10px; }
            .btn-primary { background: #4a90a4; color: white; }
            .btn-primary:hover { background: #3a7a94; }
            .btn-success { background: #28a745; color: white; }
            .btn-success:hover { background: #218838; }
            .btn-danger { background: #dc3545; color: white; }
            .btn-danger:hover { background: #c82333; }
            .message { padding: 10px; border-radius: 4px; margin-top: 10px; display: none; }
            .message.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .message.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            .current-config { background: #fff3cd; padding: 10px; border-radius: 4px; margin-bottom: 15px; }
            .test-section { margin-top: 20px; padding-top: 20px; border-top: 1px solid #ccc; }
            .test-result { background: #f8f9fa; padding: 15px; border-radius: 4px; margin-top: 10px; white-space: pre-wrap; font-family: monospace; max-height: 300px; overflow-y: auto; }
        </style>
    </head>
    <body>
        <h1>OpenAI to Claude Proxy Server</h1>
        <p class="status">&#10004; Server is running!</p>

        <div class="config-section">
            <h3>&#9881; Claude API Configuration</h3>
            <div class="current-config">
                <strong>Current Base URL:</strong> <code id="current-url">Loading...</code><br>
                <strong>API Key:</strong> <code id="current-key">Loading...</code>
            </div>

            <div class="form-group">
                <label for="base-url">Base URL:</label>
                <input type="text" id="base-url" placeholder="http://example.com/api">
            </div>

            <div class="form-group">
                <label for="api-key">API Key:</label>
                <input type="password" id="api-key" placeholder="Enter API Key">
                <button type="button" onclick="togglePassword()" style="margin-top:5px; padding:5px 10px; cursor:pointer;">Show/Hide</button>
            </div>

            <div>
                <button class="btn btn-primary" onclick="updateConfig()">Update Configuration</button>
                <button class="btn btn-success" onclick="testConnection()">Test Connection</button>
            </div>

            <div id="config-message" class="message"></div>

            <div class="test-section">
                <h4>Connection Test Result:</h4>
                <div id="test-result" class="test-result">Click "Test Connection" to verify the Claude API connection.</div>
            </div>
        </div>

        <div class="info">
            <h3>Upstream Models: <button class="btn btn-primary" style="padding:5px 10px; font-size:12px;" onclick="loadModels()">Refresh</button></h3>
            <div id="models-list">Loading models from upstream...</div>
        </div>

        <div class="info">
            <h3>Proxy Aliases:</h3>
            <p>These aliases are dynamically mapped to the latest models on server startup:</p>
            <ul>
                <li><code>tiny-model</code> → Latest Sonnet model</li>
                <li><code>bigger-model</code> → Latest Opus model</li>
                <li><code>gpt-4</code> / <code>gpt-4o</code> / <code>gpt-3.5-turbo</code> → Latest Sonnet model</li>
            </ul>
            <p id="current-mapping">Current mapping: Loading...</p>
        </div>

        <div class="info">
            <h3>Endpoints:</h3>
            <ul>
                <li><code>POST /v1/chat/completions</code> - Chat completions</li>
                <li><code>GET /v1/models</code> - List models</li>
                <li><code>GET /config</code> - Get current configuration</li>
                <li><code>POST /config</code> - Update configuration</li>
            </ul>
        </div>

        <div class="info">
            <h3>Debug Logs:</h3>
            <p>Check <code>logs/debug.log</code> for raw Claude responses</p>
            <p>Check <code>logs/error.log</code> for errors</p>
        </div>

        <script>
            // Load current config on page load
            async function loadConfig() {
                try {
                    const response = await fetch('/config');
                    const data = await response.json();
                    document.getElementById('current-url').textContent = data.base_url;
                    document.getElementById('current-key').textContent = maskApiKey(data.api_key);
                    document.getElementById('base-url').value = data.base_url;
                    document.getElementById('api-key').value = data.api_key;

                    // Display current model mapping
                    if (data.model_mapping) {
                        const mapping = data.model_mapping;
                        document.getElementById('current-mapping').innerHTML =
                            '<strong>Current mapping:</strong><br>' +
                            '<code>tiny-model</code> → ' + (mapping['tiny-model'] || 'N/A') + '<br>' +
                            '<code>bigger-model</code> → ' + (mapping['bigger-model'] || 'N/A');
                    }
                } catch (e) {
                    console.error('Failed to load config:', e);
                }
            }

            function maskApiKey(key) {
                if (!key || key.length < 10) return '***';
                return key.substring(0, 6) + '...' + key.substring(key.length - 4);
            }

            function togglePassword() {
                const input = document.getElementById('api-key');
                input.type = input.type === 'password' ? 'text' : 'password';
            }

            function showMessage(message, isError = false) {
                const el = document.getElementById('config-message');
                el.textContent = message;
                el.className = 'message ' + (isError ? 'error' : 'success');
                el.style.display = 'block';
                setTimeout(() => { el.style.display = 'none'; }, 5000);
            }

            async function updateConfig() {
                const baseUrl = document.getElementById('base-url').value.trim();
                const apiKey = document.getElementById('api-key').value.trim();

                if (!baseUrl) {
                    showMessage('Base URL is required', true);
                    return;
                }

                try {
                    const response = await fetch('/config', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ base_url: baseUrl, api_key: apiKey || undefined })
                    });

                    const data = await response.json();

                    if (response.ok) {
                        showMessage('Configuration updated successfully!');
                        document.getElementById('current-url').textContent = data.base_url;
                        document.getElementById('current-key').textContent = maskApiKey(data.api_key);
                    } else {
                        showMessage(data.error || 'Failed to update configuration', true);
                    }
                } catch (e) {
                    showMessage('Error: ' + e.message, true);
                }
            }

            async function testConnection() {
                const resultEl = document.getElementById('test-result');
                resultEl.textContent = 'Testing connection...';

                try {
                    const response = await fetch('/config/test', { method: 'POST' });
                    const data = await response.json();

                    if (data.success) {
                        resultEl.textContent = '✅ Connection successful!\\n\\n' +
                            'Response: ' + data.message + '\\n' +
                            'Model: ' + (data.model || 'N/A') + '\\n' +
                            'Latency: ' + (data.latency_ms || 'N/A') + 'ms';
                    } else {
                        resultEl.textContent = '❌ Connection failed!\\n\\nError: ' + data.error;
                    }
                } catch (e) {
                    resultEl.textContent = '❌ Test failed: ' + e.message;
                }
            }

            async function loadModels() {
                const modelsEl = document.getElementById('models-list');
                modelsEl.innerHTML = 'Loading models...';

                try {
                    const response = await fetch('/v1/models');
                    const data = await response.json();

                    if (data.data && data.data.length > 0) {
                        let html = '<table style="width:100%; border-collapse: collapse;">';
                        html += '<tr style="background:#ddd;"><th style="padding:8px; text-align:left;">Model ID</th><th style="padding:8px; text-align:left;">Owner</th></tr>';

                        data.data.forEach(model => {
                            html += '<tr><td style="padding:8px;"><code>' + model.id + '</code></td><td style="padding:8px;">' + model.owned_by + '</td></tr>';
                        });

                        html += '</table>';
                        modelsEl.innerHTML = html;
                    } else {
                        modelsEl.innerHTML = '<p style="color:red;">No models available</p>';
                    }
                } catch (e) {
                    modelsEl.innerHTML = '<p style="color:red;">Failed to load models: ' + e.message + '</p>';
                }
            }

            // Load config and models on page load
            loadConfig();
            loadModels();
        </script>
    </body>
    </html>
    """


@app.get("/config")
async def get_config():
    """获取当前配置"""
    return {
        "base_url": config.base_url,
        "api_key": config.api_key,
        "model_mapping": MODEL_MAP
    }


@app.post("/config")
async def update_config(request: Request):
    """更新配置"""
    try:
        body = await request.json()
    except:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    updated = False

    if "base_url" in body and body["base_url"]:
        config.base_url = body["base_url"].rstrip("/")
        logger.info(f"Updated base_url to: {config.base_url}")
        updated = True

    if "api_key" in body and body["api_key"]:
        config.api_key = body["api_key"]
        logger.info("Updated API key")
        updated = True

    # 保存到文件
    if updated:
        config.save()

    return {
        "success": True,
        "base_url": config.base_url,
        "api_key": config.api_key
    }


@app.post("/config/test")
async def test_config():
    """测试当前配置的连接（使用同步 requests，更稳定）"""
    import time as time_module

    # 使用已验证可用的模型
    test_model_name = MODEL_MAP.get("tiny-model", "claude-sonnet-4-20250514")

    start_time = time_module.time()

    # 使用 requests 库（和 init_model_map 一样，更稳定）
    try:
        response = requests.post(
            f"{config.base_url}/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": config.api_key,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": test_model_name,
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "Say 'Hello' in one word."}]
            },
            timeout=30
        )

        latency = int((time_module.time() - start_time) * 1000)

        if response.status_code == 200:
            data = response.json()
            content = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    content += block.get("text", "")

            return {
                "success": True,
                "message": content[:100] if content else "OK",
                "model": data.get("model", "unknown"),
                "latency_ms": latency
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text[:200]}"
            }

    except requests.RequestException as e:
        return {
            "success": False,
            "error": f"Connection error: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error: {str(e)}"
        }


@app.get("/v1/models")
async def list_models():
    """从上游API获取真实的模型列表"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{config.base_url}/v1/models",
                headers={
                    "x-api-key": config.api_key,
                    "anthropic-version": "2023-06-01"
                }
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get models from upstream: {response.status_code}")
    except Exception as e:
        logger.error(f"Error fetching models: {e}")

    # 如果上游失败，返回空列表
    return {"object": "list", "data": []}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Invalid JSON: {str(e)}", "type": "invalid_request_error"}}
        )

    openai_model = body.get("model", "claude-sonnet-4-20250514")
    stream = body.get("stream", False)
    messages = body.get("messages", [])
    tools = body.get("tools", [])
    tool_choice = body.get("tool_choice")

    # 保存用户消息
    save_user_message(messages)

    # 转换消息格式
    system_prompt, claude_messages = await convert_openai_to_claude(messages)

    # 确定 Claude 模型 (默认使用 opus)
    if openai_model in MODEL_MAP:
        claude_model = MODEL_MAP[openai_model]
    elif "claude" in openai_model:
        claude_model = openai_model
    else:
        claude_model = "claude-sonnet-4-5-20250929"

    # 构建 Claude API 请求
    claude_payload = {
        "model": claude_model,
        "max_tokens": body.get("max_tokens") or body.get("max_completion_tokens") or 8192,
        "messages": claude_messages,
        "stream": stream
    }

    if system_prompt:
        claude_payload["system"] = system_prompt

    # 添加可选参数
    if "temperature" in body:
        temp = body["temperature"]
        claude_payload["temperature"] = min(temp, 1.0)

    if "top_p" in body:
        claude_payload["top_p"] = body["top_p"]

    if "stop" in body:
        stop = body["stop"]
        if isinstance(stop, str):
            claude_payload["stop_sequences"] = [stop]
        elif isinstance(stop, list):
            claude_payload["stop_sequences"] = stop

    # 添加工具
    if tools:
        claude_tools = convert_openai_tools_to_claude(tools)
        if claude_tools:
            claude_payload["tools"] = claude_tools

        claude_tool_choice = convert_openai_tool_choice_to_claude(tool_choice)
        if claude_tool_choice:
            claude_payload["tool_choice"] = claude_tool_choice

    headers = {
        "Content-Type": "application/json",
        "x-api-key": config.api_key,
        "anthropic-version": "2023-06-01"
    }

    claude_url = f"{config.base_url}/v1/messages"

    logger.info(f"Proxying request to {claude_url}, model: {claude_model}, stream: {stream}")
    logger.debug(f"Claude payload: {json.dumps(claude_payload, ensure_ascii=False)[:1000]}")

    if stream:
        return StreamingResponse(
            stream_claude_response(claude_url, headers, claude_payload, openai_model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    # 非流式请求 - 带重试机制
    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        if attempt > 0:
            wait_time = 2 ** attempt
            logger.info(f"Non-stream retry attempt {attempt + 1}/{max_retries} after {wait_time}s wait")
            await asyncio.sleep(wait_time)

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(claude_url, headers=headers, json=claude_payload)

                if response.status_code != 200:
                    error_body = response.text
                    logger.error(f"Claude API error (attempt {attempt + 1}): {response.status_code} - {error_body}")

                    # 检查是否为可重试的错误
                    if response.status_code in (500, 502, 503, 504):
                        if "ReadableStream" in error_body or "unavailable" in error_body.lower():
                            last_error = error_body
                            continue  # 重试

                    return JSONResponse(
                        status_code=response.status_code,
                        content={
                            "error": {
                                "message": error_body,
                                "type": "api_error",
                                "code": response.status_code
                            }
                        }
                    )

                claude_response = response.json()
                logger.debug(f"Claude response: {json.dumps(claude_response, ensure_ascii=False)[:2000]}")
                return convert_claude_to_openai(claude_response, openai_model)

        except httpx.RequestError as e:
            logger.error(f"Request error (attempt {attempt + 1}): {str(e)}")
            last_error = str(e)
            continue  # 重试

    # 所有重试都失败了
    logger.error(f"All {max_retries} non-stream attempts failed. Last error: {last_error}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": f"Failed after {max_retries} retries: {last_error}",
                "type": "connection_error"
            }
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
