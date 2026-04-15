#!/usr/bin/env python3
"""
实时语音对话 Web 服务
- 浏览器录音 → Whisper STT → Hermes Chat → Edge TTS → 浏览器播放
"""

import asyncio
import json
import os
import subprocess
import tempfile
import uuid
import wave
import io
import struct
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# 音频缓存目录
AUDIO_DIR = Path.home() / ".hermes" / "audio_cache"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def pcm_to_wav(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1, bits: int = 16) -> bytes:
    """将PCM数据转为WAV格式"""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(bits // 8)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


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
    """用本地 Whisper CLI 做语音识别"""
    proc = await asyncio.create_subprocess_exec(
        "whisper", audio_path,
        "--model", "base",
        "--language", "zh",
        "--output_format", "txt",
        "--output_dir", "/tmp/whisper_out",
        "--verbose", "False",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await asyncio.wait_for(proc.communicate(), timeout=30)
    # whisper 输出到 /tmp/whisper_out/xxx.txt
    basename = Path(audio_path).stem
    txt_path = Path(f"/tmp/whisper_out/{basename}.txt")
    if txt_path.exists():
        text = txt_path.read_text().strip()
        txt_path.unlink()
        return text if text else "（语音识别为空，请重试）"
    return "（语音识别失败，请重试）"


async def ask_hermes(query: str, session_id: str = "") -> tuple[str, str]:
    """调用 hermes chat -q 获取回复和音频"""
    cmd = ["hermes", "chat", "-q", query, "--quiet"]
    if session_id:
        cmd.extend(["--resume", session_id])

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "NO_COLOR": "1"},
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
    output = stdout.decode().strip()

    # 解析输出，提取 MEDIA 路径和文本
    audio_path = ""
    text_lines = []
    new_session_id = ""

    for line in output.split("\n"):
        line = line.strip()
        if line.startswith("MEDIA:"):
            audio_path = line.replace("MEDIA:", "").strip()
        elif line.startswith("session_id:"):
            new_session_id = line.replace("session_id:", "").strip()
        elif line and not line.startswith("网络链路") and not line.startswith("DNS"):
            text_lines.append(line)

    text = "\n".join(text_lines) if text_lines else output
    return text, audio_path, new_session_id


@app.get("/")
async def index():
    """返回前端页面"""
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text())


@app.websocket("/ws/voice")
async def voice_chat(websocket: WebSocket):
    """WebSocket 语音对话"""
    await websocket.accept()
    session_id = ""

    await websocket.send_json({"type": "status", "message": "已连接，请按住说话..."})

    try:
        while True:
            msg = await websocket.receive()

            if "bytes" in msg:
                # 收到音频数据（webm/opus片段）
                audio_data = msg["bytes"]
                if len(audio_data) < 100:
                    continue

                await websocket.send_json({"type": "status", "message": "正在识别语音..."})

                # 保存为临时文件
                tmp_id = uuid.uuid4().hex[:8]
                tmp_webm = f"/tmp/voice_{tmp_id}.webm"
                with open(tmp_webm, 'wb') as f:
                    f.write(audio_data)

                try:
                    # ffmpeg 转 wav
                    wav_path = await convert_to_wav(tmp_webm)

                    # STT
                    text = await transcribe_audio(wav_path)
                    await websocket.send_json({"type": "user_text", "message": text})

                    if "失败" in text:
                        await websocket.send_json({"type": "status", "message": "识别失败，请重试"})
                        continue

                    # Hermes 回复
                    await websocket.send_json({"type": "status", "message": "正在思考..."})
                    reply_text, audio_path, new_session_id = await ask_hermes(text, session_id)

                    if new_session_id:
                        session_id = new_session_id

                    # 发送文本回复
                    await websocket.send_json({"type": "assistant_text", "message": reply_text})

                    # 发送音频
                    if audio_path and os.path.exists(audio_path):
                        audio_data = Path(audio_path).read_bytes()
                        await websocket.send_json({
                            "type": "audio",
                            "format": "ogg",
                            "size": len(audio_data),
                        })
                        # 分块发送音频
                        chunk_size = 8192
                        for i in range(0, len(audio_data), chunk_size):
                            await websocket.send_bytes(audio_data[i:i+chunk_size])
                        await websocket.send_json({"type": "audio_end"})

                    await websocket.send_json({"type": "status", "message": "按住说话..."})

                except asyncio.TimeoutError:
                    await websocket.send_json({"type": "status", "message": "超时，请重试"})
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": str(e)})
                finally:
                    # 清理临时文件
                    for f in [tmp_webm, wav_path, f"/tmp/whisper_out/voice_{tmp_id}_16k.txt"]:
                        try: os.unlink(f)
                        except: pass

            elif "text" in msg:
                # 收到文本消息
                try:
                    data = json.loads(msg["text"])
                    if data.get("type") == "text_input":
                        text = data.get("message", "").strip()
                        if text:
                            await websocket.send_json({"type": "status", "message": "正在思考..."})
                            reply_text, audio_path, new_session_id = await ask_hermes(text, session_id)
                            if new_session_id:
                                session_id = new_session_id
                            await websocket.send_json({"type": "assistant_text", "message": reply_text})
                            if audio_path and os.path.exists(audio_path):
                                audio_data = Path(audio_path).read_bytes()
                                await websocket.send_json({"type": "audio", "format": "ogg", "size": len(audio_data)})
                                chunk_size = 8192
                                for i in range(0, len(audio_data), chunk_size):
                                    await websocket.send_bytes(audio_data[i:i+chunk_size])
                                await websocket.send_json({"type": "audio_end"})
                            await websocket.send_json({"type": "status", "message": "按住说话..."})
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
