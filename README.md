# Hermes Voice Chat - 实时语音对话助手

通过浏览器与 Hermes Agent 进行实时语音对话的 Web 应用。

## 功能特性

- 按住麦克风说话，松开发送
- 支持文字输入模式
- 实时语音识别（本地 Whisper）
- AI 回复自动语音播放（Edge TTS）
- WebSocket 实时双向通信
- 手机/电脑浏览器均可使用

## 架构流程

```
浏览器录音(webm) 
    → WebSocket 发送 
    → FastAPI 后端接收 
    → ffmpeg 转 wav 
    → Whisper 语音识别 
    → Hermes Chat API 
    → Edge TTS 语音合成 
    → WebSocket 返回音频 
    → 浏览器播放
```

## 环境要求

- macOS / Linux
- Python 3.9+
- ffmpeg（音频格式转换）
- Whisper（语音识别）
- Hermes Agent（AI 对话引擎）

## 安装依赖

```bash
# Python 依赖
pip3 install fastapi uvicorn websockets

# ffmpeg（macOS）
brew install ffmpeg

# Whisper（OpenAI 开源语音识别）
pip3 install openai-whisper

# 确保 Hermes Agent 已安装
hermes --version
```

## 快速开始

```bash
# 克隆仓库
git clone git@github.com:dockercore/Hermes-Voice-Chat.git
cd Hermes-Voice-Chat

# 启动服务
python3 server.py
```

打开浏览器访问: http://localhost:8765

## 使用方式

1. 打开页面后，状态指示灯变绿表示已连接
2. 按住麦克风按钮说话，松开发送
3. 也可以在底部输入框直接输入文字
4. AI 回复会自动以语音形式播放

## 配置说明

### TTS 语音（hermes 配置）

在 `~/.hermes/config.yaml` 中配置：

```yaml
tts:
  provider: edge
  edge:
    voice: zh-CN-XiaoxiaoNeural  # 中文女声
```

可选语音：
- `zh-CN-XiaoxiaoNeural` - 晓晓（女声，推荐）
- `zh-CN-YunxiNeural` - 云希（男声）
- `zh-CN-XiaoyiNeural` - 晓伊（女声，活泼）

### STT 语音识别

默认使用本地 Whisper `base` 模型，支持中文识别。
可在 `server.py` 中修改 `--model` 参数切换模型：

| 模型 | 大小 | 速度 | 精度 |
|------|------|------|------|
| tiny | 39M | 最快 | 一般 |
| base | 74M | 快 | 较好 |
| small | 244M | 中等 | 好 |
| medium | 769M | 较慢 | 很好 |

### 端口配置

默认端口 `8765`，可在 `server.py` 末尾修改：

```python
uvicorn.run(app, host="0.0.0.0", port=8765)
```

## 外网访问（可选）

使用 Cloudflare Tunnel 暴露到外网：

```bash
cloudflared tunnel --url http://localhost:8765
```

会生成一个 `https://xxx.trycloudflare.com` 地址，可从任何设备访问。

## 文件结构

```
Hermes-Voice-Chat/
├── server.py        # FastAPI 后端服务
├── index.html       # 前端页面（语音UI）
├── README.md        # 本文档
└── LICENSE          # MIT 许可证
```

## 技术栈

- **后端**: FastAPI + WebSocket + uvicorn
- **前端**: 原生 HTML/CSS/JS + Web Audio API
- **语音识别**: OpenAI Whisper（本地）
- **语音合成**: Microsoft Edge TTS（通过 Hermes）
- **音频处理**: ffmpeg

## 许可证

MIT License

## 致谢

- [Hermes Agent](https://github.com/h Hermes Agent) - AI 助手引擎
- [OpenAI Whisper](https://github.com/openai/whisper) - 语音识别
- [FastAPI](https://fastapi.tiangolo.com/) - Web 框架
