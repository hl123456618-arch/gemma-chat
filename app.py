import os, json, datetime, math, re, uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-8b-8192"

SYSTEM_PROMPT = """You are Gemma, an advanced AI assistant that is brilliant, creative, warm, and deeply helpful.

Your personality:
- Friendly and enthusiastic — you genuinely enjoy helping people
- Intellectually curious — you go deep on topics when it's useful
- Clear and structured — use markdown, headers, bullets, and code blocks to make responses easy to read
- Proactive — suggest follow-ups, point out related insights the user might not have considered
- Honest — clearly state when you're uncertain, and use your tools to verify facts

Tool usage — be proactive:
- ALWAYS use get_current_time when asked about time, date, day, or age calculations
- ALWAYS use search_web when asked about current events, news, people, companies, prices, or anything that changes over time
- ALWAYS use get_weather for any weather or forecast questions
- ALWAYS use calculate for math — never compute in your head when precision matters
- Use get_news to find recent news on any topic

Format your responses beautifully with markdown. Use **bold** for key terms, code blocks for code, and structured lists when presenting multiple items."""

# ─── TOOLS ───────────────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date, time, day of week, and timezone",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for any information — facts, news, people, companies, products, definitions",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather conditions and forecast for any location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name, region, or coordinates"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate any mathematical expression with full precision",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Mathematical expression (e.g. '2 ** 32', 'sqrt(144)', '15% of 3500')"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Get recent news headlines on any topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "News topic or keyword"}
                },
                "required": ["topic"]
            }
        }
    }
]

# ─── TOOL IMPLEMENTATIONS ─────────────────────────────────────────────────────
async def get_current_time():
    now = datetime.datetime.now()
    return {
        "datetime": now.strftime("%A, %B %d, %Y at %I:%M:%S %p"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
        "iso": now.isoformat(),
        "unix": int(now.timestamp())
    }

async def search_web(query: str):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"https://api.duckduckgo.com/?q={query}&format=json&no_redirect=1&no_html=1&skip_disambig=1",
                headers={"User-Agent": "GemmaAI/1.0"}
            )
            d = r.json()
            result = {
                "answer": d.get("AbstractText") or d.get("Answer") or "",
                "source": d.get("AbstractSource", ""),
                "url": d.get("AbstractURL", ""),
                "related": [t.get("Text", "") for t in d.get("RelatedTopics", [])[:5] if t.get("Text")]
            }
            return result if result["answer"] else {"message": "No instant answer found — try a more specific query.", "query": query, "related": result["related"]}
    except Exception as e:
        return {"error": str(e), "query": query}

async def get_weather(location: str):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"https://wttr.in/{location}?format=j1", headers={"User-Agent": "GemmaAI/1.0"})
            d = r.json()
            c = d["current_condition"][0]
            area = d.get("nearest_area", [{}])[0]
            forecast = []
            for day in d.get("weather", [])[:3]:
                forecast.append({
                    "date": day.get("date"),
                    "max_c": day.get("maxtempC"),
                    "min_c": day.get("mintempC"),
                    "description": day.get("hourly", [{}])[4].get("weatherDesc", [{}])[0].get("value", "")
                })
            return {
                "location": area.get("areaName", [{}])[0].get("value", location),
                "country": area.get("country", [{}])[0].get("value", ""),
                "temp_c": c["temp_C"], "temp_f": c["temp_F"],
                "feels_like_c": c["FeelsLikeC"], "feels_like_f": c["FeelsLikeF"],
                "description": c["weatherDesc"][0]["value"],
                "humidity": c["humidity"] + "%",
                "wind_kmph": c["windspeedKmph"],
                "wind_direction": c["winddir16Point"],
                "visibility_km": c["visibility"],
                "uv_index": c["uvIndex"],
                "3_day_forecast": forecast
            }
    except Exception as e:
        return {"error": str(e), "location": location}

async def calculate(expression: str):
    try:
        # Handle percentage expressions like "15% of 3500"
        expression = re.sub(r'(\d+(?:\.\d+)?)%\s+of\s+(\d+(?:\.\d+)?)', r'(\1/100)*\2', expression, flags=re.IGNORECASE)
        safe = re.sub(r'[^0-9+\-*/.()%^ eE]', '', expression)
        safe_globals = {"__builtins__": {}}
        safe_locals = {
            "abs": abs, "round": round, "pow": pow, "min": min, "max": max,
            "sqrt": math.sqrt, "pi": math.pi, "e": math.e,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "log": math.log, "log10": math.log10, "floor": math.floor, "ceil": math.ceil
        }
        result = eval(safe, safe_globals, safe_locals)
        return {"expression": expression, "result": result, "formatted": f"{result:,}" if isinstance(result, (int, float)) else str(result)}
    except Exception as e:
        return {"error": str(e), "expression": expression}

async def get_news(topic: str):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"https://api.duckduckgo.com/?q={topic}+news&format=json&no_redirect=1&no_html=1",
                headers={"User-Agent": "GemmaAI/1.0"}
            )
            d = r.json()
            topics = [t.get("Text", "") for t in d.get("RelatedTopics", [])[:6] if t.get("Text")]
            answer = d.get("AbstractText", "")
            return {"topic": topic, "summary": answer, "headlines": topics}
    except Exception as e:
        return {"error": str(e), "topic": topic}

TOOL_MAP = {
    "get_current_time": get_current_time,
    "search_web": search_web,
    "get_weather": get_weather,
    "calculate": calculate,
    "get_news": get_news
}

# ─── ROUTES ───────────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

class ChatRequest(BaseModel):
    messages: list[dict]

@app.post("/chat")
async def chat(req: ChatRequest):
    async def stream():
        allowed = {'role', 'content', 'tool_calls', 'tool_call_id', 'name'}
        clean = [{k: v for k, v in m.items() if k in allowed} for m in req.messages]
        history = [{"role": "system", "content": SYSTEM_PROMPT}] + clean

        # Tool-call turns: non-streaming so we can detect tool_calls
        for turn in range(5):
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    r = await client.post(
                        GROQ_URL,
                        json={"model": MODEL, "messages": history, "tools": TOOLS, "stream": False, "max_tokens": 2048},
                        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
                    )
                    data = r.json()
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                return

            msg = data.get("choices", [{}])[0].get("message", {})
            if not msg:
                err = data.get("error", {}).get("message", "No response from API")
                yield f"data: {json.dumps({'type': 'error', 'message': err})}\n\n"
                return

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                # No tools needed — re-request with streaming for fast first token
                break

            history.append(msg)
            for call in tool_calls:
                fn_name = call["function"]["name"]
                raw_args = call["function"].get("arguments", "{}")
                fn_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                yield f"data: {json.dumps({'type': 'tool_call', 'name': fn_name, 'args': fn_args})}\n\n"
                fn = TOOL_MAP.get(fn_name)
                result = await fn(**fn_args) if fn else {"error": f"Unknown tool: {fn_name}"}
                yield f"data: {json.dumps({'type': 'tool_result', 'name': fn_name, 'result': result})}\n\n"
                history.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})

        # Final streaming response — tokens appear immediately
        try:
            yield f"data: {json.dumps({'type': 'stream_start'})}\n\n"
            async with httpx.AsyncClient(timeout=60) as client:
                async with client.stream(
                    "POST", GROQ_URL,
                    json={"model": MODEL, "messages": history, "stream": True, "max_tokens": 2048},
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
                ) as r:
                    async for line in r.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                            token = chunk["choices"][0]["delta"].get("content", "")
                            if token:
                                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                        except Exception:
                            pass
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}
