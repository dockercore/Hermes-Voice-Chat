#!/usr/bin/env python3
"""
实时语音对话 Web 服务 v3
- 浏览器录音 → faster-whisper STT → Hermes Agent 执行任务 → Edge TTS → 浏览器播放
- Hermes 拥有完整工具链：终端、浏览器、文件、搜索、记忆等
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

# Hermes 执行超时（秒）
HERMES_TIMEOUT = 120

# ===== faster-whisper 常驻模型 =====
whisper_model = None

def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        from faster_whisper import WhisperModel
        print("加载 faster-whisper 模型...")
        whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        print("faster-whisper 模型加载完成！")
    return whisper_model


async def convert_to_wav(input_path: str) -> str:
    """用 ffmpeg 将任意音频格式转为 16kHz mono WAV"""
    output_path = input_path.rsplit('.', 1)[0] + '_16k.wav'
    proc = await asyncio.create_subprocess_exec(
        'ffmpeg', '-y', '-i', input_path,
        '-ar', '16000', '-ac', '1', '-f', 'wav', output_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    return output_path if Path(output_path).exists() else input_path


async def transcribe_audio(audio_path: str) -> str:
    """faster-whisper 语音识别"""
    t0 = time.time()
    model = get_whisper_model()
    loop = asyncio.get_event_loop()
    segments, info = await loop.run_in_executor(
        None,
        lambda: model.transcribe(audio_path, language="zh", beam_size=5)
    )
    text = "".join(seg.text for seg in segments).strip()
    elapsed = time.time() - t0
    print(f"[STT] {elapsed:.2f}s | {text}")
    return text if text else "（识别为空，请重试）"


async def ask_hermes(query: str, websocket: WebSocket = None) -> str:
    """调用 Hermes Agent 执行任务，返回结果文本"""
    t0 = time.time()
    
    # 使用 hermes chat -q 一次性执行
    proc = await asyncio.create_subprocess_exec(
        'hermes', 'chat', '-q', query, '--quiet',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), 
            timeout=HERMES_TIMEOUT
        )
    except asyncio.TimeoutError:
        proc.kill()
        return "Hermes 执行超时，请简化你的问题或稍后再试。"
    
    elapsed = time.time() - t0
    
    # 解析输出
    output = stdout.decode('utf-8', errors='replace').strip()
    err = stderr.decode('utf-8', errors='replace').strip()
    
    if proc.returncode != 0 and not output:
        print(f"[Hermes] ERROR ({elapsed:.1f}s) | {err[:200]}")
        return f"执行出错：{err[:200]}"
    
    # Hermes 输出可能包含工具调用日志，提取最终回复
    # 通常最终回复在最后部分
    result = output if output else "（无输出）"
    
    # 如果输出太长，截取最后3000字符（TTS有限制）
    if len(result) > 3000:
        result = "..." + result[-3000:]
    
    print(f"[Hermes] {elapsed:.1f}s | {result[:100]}...")
    return result


async def text_to_speech(text: str) -> str:
    """Edge TTS 生成语音"""
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
    history = []

    await websocket.send_json({"type": "status", "message": "已连接，请按住说话..."})

    try:
        while True:
            msg = await websocket.receive()

            if "bytes" in msg:
                audio_data = msg["bytes"]
                if len(audio_data) < 100:
                    continue

                await websocket.send_json({"type": "status", "message": "识别语音..."})

                tmp_id = uuid.uuid4().hex[:8]
                tmp_webm = f"/tmp/voice_{tmp_id}.webm"
                with open(tmp_webm, 'wb') as f:
                    f.write(audio_data)

                try:
                    # STT
                    wav_path = await convert_to_wav(tmp_webm)
                    text = await transcribe_audio(wav_path)
                    await websocket.send_json({"type": "user_text", "message": text})

                    if "失败" in text or "空" in text:
                        await websocket.send_json({"type": "status", "message": "识别失败，请重试"})
                        continue

                    # Hermes 执行任务
                    await websocket.send_json({"type": "status", "message": "Hermes 正在执行任务..."})
                    reply = await ask_hermes(text, websocket)

                    await websocket.send_json({"type": "assistant_text", "message": reply})

                    # TTS
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
                    for f in [tmp_webm, wav_path]:
                        try: os.unlink(f)
                        except: pass

            elif "text" in msg:
                try:
                    data = json.loads(msg["text"])
                    if data.get("type") == "text_input":
                        text = data.get("message", "").strip()
                        if text:
                            await websocket.send_json({"type": "status", "message": "Hermes 正在执行任务..."})
                            reply = await ask_hermes(text, websocket)

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

    except WebSocketDisconnect:
        print("[WS] 客户端断开")
    except Exception as e:
        print(f"[WS] 异常: {e}")


@app.on_event("startup")
async def startup():
    get_whisper_model()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
