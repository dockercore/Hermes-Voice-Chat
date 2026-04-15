#!/bin/bash
# 启动语音对话服务
cd /Users/dockercore/voice-chat
source venv/bin/activate
export GROQ_API_KEY="REDACTED"
exec python3 server.py
