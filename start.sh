#!/bin/bash
# 启动语音对话服务
# 使用前请设置环境变量: export GROQ_API_KEY="your-key-here"
cd /Users/dockercore/voice-chat
source venv/bin/activate
exec python3 server.py
