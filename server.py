#!/usr/bin/env python3
"""
实时语音对话 Web 服务 v4
- 浏览器录音 → faster-whisper STT → Hermes Agent 执行任务 → Edge TTS → 浏览器播放
- 支持随时打断：按住麦克风即可中断当前任务
"""

import asyncio
import json
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI()

# ===== 配置 =====
AUDIO_DIR = Path.home() / ".hermes" / "audio_cache"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

HERMES_TIMEOUT = 120

# Groq Whisper API
GROQ_API_KEY = "REDACTED"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


async def transcribe_audio(audio_path: str) -> str:
    """Groq Whisper API 语音识别"""
    t0 = time.time()
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

    with open(audio_path, "rb") as f:
        resp = await client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=f,
            language="zh",
        )

    text = resp.text.strip()
    elapsed = time.time() - t0
    print(f"[STT/Groq] {elapsed:.2f}s | {text}")
    return text if text else "（识别为空，请重试）"


async def ask_hermes(query: str, cancel_event: asyncio.Event) -> str:
    """调用 Hermes Agent，支持通过 cancel_event 打断"""
    t0 = time.time()

    proc = await asyncio.create_subprocess_exec(
        'hermes', 'chat', '-q', query, '--quiet',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        # 同时等待完成或取消
        comm_task = asyncio.create_task(proc.communicate())
        cancel_task = asyncio.create_task(cancel_event.wait())

        done, pending = await asyncio.wait(
            {comm_task, cancel_task},
            timeout=HERMES_TIMEOUT,
            return_when=asyncio.FIRST_COMPLETED,
        )

        # 取消没完成的
        for t in pending:
            t.cancel()

        if cancel_task in done:
            # 用户打断了
            proc.kill()
            await proc.wait()
            print(f"[Hermes] INTERRUPTED ({time.time()-t0:.1f}s)")
            return "__INTERRUPTED__"

        if comm_task not in done:
            # 超时
            proc.kill()
            await proc.wait()
            return "Hermes 执行超时，请简化你的问题或稍后再试。"

        stdout, stderr = comm_task.result()

    except Exception as e:
        proc.kill()
        return f"执行出错：{e}"

    elapsed = time.time() - t0
    output = stdout.decode('utf-8', errors='replace').strip()

    if proc.returncode != 0 and not output:
        err = stderr.decode('utf-8', errors='replace').strip()
        print(f"[Hermes] ERROR ({elapsed:.1f}s) | {err[:200]}")
        return f"执行出错：{err[:200]}"

    result = output if output else "（无输出）"
    if len(result) > 3000:
        result = "..." + result[-3000:]

    print(f"[Hermes] {elapsed:.1f}s | {result[:100]}...")
    return result


async def text_to_speech(text: str) -> str:
    t0 = time.time()
    import edge_tts
    output_path = str(AUDIO_DIR / f"tts_{uuid.uuid4().hex[:8]}.ogg")
    communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
    await communicate.save(output_path)
    elapsed = time.time() - t0
    print(f"[TTS] {elapsed:.2f}s | {output_path}")
    return output_path


# ===== 路由 =====

@app.get("/")
async def index():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text())


@app.websocket("/ws/voice")
async def voice_chat(websocket: WebSocket):
    await websocket.accept()

    # 每个连接有自己的取消事件
    cancel_event = asyncio.Event()

    await websocket.send_json({"type": "status", "message": "已连接，请按住说话..."})

    try:
        while True:
            msg = await websocket.receive()

            # 处理打断请求
            if "text" in msg:
                try:
                    data = json.loads(msg["text"])
                    if data.get("type") == "interrupt":
                        cancel_event.set()
                        cancel_event.clear()
                        print("[WS] 收到打断请求")
                        await websocket.send_json({"type": "interrupted"})
                        continue

                    if data.get("type") == "text_input":
                        text = data.get("message", "").strip()
                        if text:
                            cancel_event.clear()
                            await websocket.send_json({"type": "status", "message": "Hermes 正在执行任务..."})
                            reply = await ask_hermes(text, cancel_event)

                            if reply == "__INTERRUPTED__":
                                await websocket.send_json({"type": "status", "message": "已打断，请说话..."})
                                continue

                            await websocket.send_json({"type": "assistant_text", "message": reply})

                            audio_path = await text_to_speech(reply)
                            if os.path.exists(audio_path):
                                audio_bytes = Path(audio_path).read_bytes()
                                await websocket.send_json({"type": "audio", "format": "ogg", "size": len(audio_bytes)})
                                chunk_size = 16384
                                for i in range(0, len(audio_bytes), chunk_size):
                                    await websocket.send_bytes(audio_bytes[i:i+chunk_size])
                                await websocket.send_json({"type": "audio_end"})

                            await websocket.send_json({"type": "status", "message": "按住说话..."})
                except Exception as e:
                    print(f"[ERROR] {e}")
                    try:
                        await websocket.send_json({"type": "error", "message": str(e)})
                    except:
                        pass

            elif "bytes" in msg:
                audio_data = msg["bytes"]
                if len(audio_data) < 100:
                    continue

                await websocket.send_json({"type": "status", "message": "识别语音..."})

                tmp_id = uuid.uuid4().hex[:8]
                tmp_webm = f"/tmp/voice_{tmp_id}.webm"
                with open(tmp_webm, 'wb') as f:
                    f.write(audio_data)
                try:
                    # STT - Groq 直接支持 webm
                    text = await transcribe_audio(tmp_webm)
                    await websocket.send_json({"type": "user_text", "message": text})

                    if "失败" in text or "空" in text:
                        await websocket.send_json({"type": "status", "message": "识别失败，请重试"})
                        continue

                    cancel_event.clear()
                    await websocket.send_json({"type": "status", "message": "Hermes 正在执行任务..."})
                    reply = await ask_hermes(text, cancel_event)

                    if reply == "__INTERRUPTED__":
                        await websocket.send_json({"type": "status", "message": "已打断，请说话..."})
                        continue

                    await websocket.send_json({"type": "assistant_text", "message": reply})

                    await websocket.send_json({"type": "status", "message": "生成语音..."})
                    audio_path = await text_to_speech(reply)

                    if os.path.exists(audio_path):
                        audio_bytes = Path(audio_path).read_bytes()
                        await websocket.send_json({
                            "type": "audio",
                            "format": "ogg",
                            "size": len(audio_bytes),
                        })
                        chunk_size = 16384
                        for i in range(0, len(audio_bytes), chunk_size):
                            await websocket.send_bytes(audio_bytes[i:i+chunk_size])
                        await websocket.send_json({"type": "audio_end"})

                    await websocket.send_json({"type": "status", "message": "按住说话..."})

                except asyncio.TimeoutError:
                    await websocket.send_json({"type": "status", "message": "超时，请重试"})
                except Exception as e:
                    print(f"[ERROR] {e}")
                    try:
                        await websocket.send_json({"type": "error", "message": str(e)})
                    except:
                        pass
                finally:
                    try: os.unlink(tmp_webm)
                    except: pass

    except WebSocketDisconnect:
        print("[WS] 客户端断开")
    except Exception as e:
        print(f"[WS] 异常: {e}")


@app.on_event("startup")
async def startup():
    print("Groq Whisper API 就绪，无需加载本地模型")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
