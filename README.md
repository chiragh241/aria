# Aria - Personal AI Assistant

A fully-featured personal AI assistant that runs locally with hybrid LLM support (local Ollama + cloud Claude), multi-channel messaging (Slack, WhatsApp, Web), configurable security profiles, and extensible skills.

## Features

- **Hybrid LLM**: Routes simple tasks to local Ollama, complex reasoning to Claude API
- **Smart Ollama Setup**: Hardware-detection during onboarding suggests the best model for your machine; can download it for you
- **Multi-Channel**: Slack, WhatsApp (via web bridge), and Web UI
- **Security**: Configurable profiles (paranoid, balanced, trusted) with approval workflows
- **Self-Healing**: Monitors logs for errors, auto-fixes via patterns (pip install, etc.) or LLM suggestions; code edits require your approval in the Approvals UI with diff preview
- **Sandboxed Execution**: Docker isolation for untrusted code
- **Memory**: Short-term conversation, long-term vector store (ChromaDB), episodic task history, user profiles
- **Knowledge Graph**: Cognee integration for entity/relationship extraction—synced with user profiles
- **29 Features**: Morning briefing, context reminders, proactive suggestions, time-of-day awareness, cost/usage tracking, theme toggle, data export, PWA, push notifications, keyboard shortcuts, debug trace, skill templates, and more
- **Integrations**: Notion, Todoist, Linear, Spotify (API keys configurable in Settings or onboarding)
- **Built-in Skills**: File ops, shell, browser, calendar, email, SMS, TTS, STT, image, video, documents, weather, research, agent, memory, finance, news, contacts, tracking, webhook, home, notion, todoist, linear, spotify
- **Dynamic Skills**: AI can generate new skills on demand
- **OneContext**: Unified context across all channels and agents — share and load context so anyone on web, Slack, or WhatsApp can continue from the same point ([OneContext](https://github.com/TheAgentContextLab/OneContext)). Enabled by default.
- **Web Dashboard**: Chat, approvals, settings, logs, skills, setup onboarding, cost/usage widgets, data export
- **Slash Commands**: `/help`, `/clear`, `/status`, `/skills`, `/capabilities` in web chat and all channels
- **"What can you do"**: Ask in plain language for a detailed list of skills and how to trigger them

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ (for WhatsApp bridge)
- Docker (optional, for sandboxing)
- Ollama (optional, for local LLM)

### Installation

1. **Clone and install dependencies**:
   ```bash
   cd aria
   pip install uv  # or use pip install -e .
   uv pip install -e .
   ```

2. **Run setup wizard** (recommended):
   ```bash
   python -m src.main --setup
   ```
   The wizard configures LLM providers, channels, skills, and integrations (Notion, Todoist, Linear, Spotify). When you choose Ollama, it detects your hardware (RAM, GPU) and suggests the best model. Only models that support tool/function calling are recommended so skills work.

3. **Set up environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Install frontend dependencies**:
   ```bash
   cd src/web/frontend
   npm install
   ```

5. **(Optional) Set up WhatsApp bridge** (if enabled in setup):
   ```bash
   cd whatsapp-bridge
   npm install
   ```

6. **(Optional) Ollama models**: If you skipped the setup wizard, pull a tool-capable model (skills require function calling):
   ```bash
   ollama pull llama3.2:8b
   ```

7. **(Optional) Mic/voice transcription**: Requires `openai-whisper` and `ffmpeg`:
   ```bash
   pip install openai-whisper
   # ffmpeg: brew install ffmpeg (Mac) or apt install ffmpeg (Linux)
   ```
   First mic use downloads the Whisper model (~150MB for "base").

### Troubleshooting

- **"Transcription failed"**: Install `openai-whisper` and `ffmpeg`. Grant browser mic permission.
- **"ChromaDB not installed"**: Reinstall dependencies: `pip install -e .` or `uv pip install -e .`
- **vite.svg 404**: Rebuild the frontend: `cd src/web/frontend && npm run build`
- **Self-healing not detecting errors**: It reads `data/logs/aria.log` every 10s. Terminal-only output (e.g. uvicorn access logs) may not be in the log file. Ask "check logs and fix" to trigger immediately.
- **Self-healing code fixes**: When the LLM suggests a code fix, it appears in **Approvals** (dashboard). Review the file name and old/new diff, then Approve or Deny. Only approved edits are applied.
- **OneContext (unified agent context)**: Install `npm i -g onecontext-ai` and enable `onecontext.enabled: true` in config. Use `context_skill.share_context` and `context_skill.load_context` to share/load context. All agents (research, coding, data) share the same context.

### Running

**Development mode**:
```bash
# Terminal 1: Start backend
python -m src.main

# Hot reload on 8080: run with --dev (Vite on 8080, backend on 8081)
python -m src.main --dev

# After frontend changes without --dev: rebuild
cd src/web/frontend && npm run build
```

**With Docker**:
```bash
cd docker
docker-compose up -d
```

### Configuration

Edit `config/settings.yaml` and `config/self_healing.yaml` to customize:
- LLM providers and models
- Channel settings (Slack, WhatsApp, Web)
- Security profile
- Memory, knowledge graph (Cognee), and sandbox options
- **Self-healing** (`config/self_healing.yaml`): patterns for known errors, pip packages, LLM fallback (including `code_edit` with approval)
- Skills and integrations (Notion, Todoist, Linear, Spotify)

**Web UI**: After login, use **Setup** (onboarding) or **Settings** to select skills, add integration API keys, and configure the system.

### Web Chat Commands

In the web chat (and Slack/WhatsApp), you can use:
- `/help` — Show available commands
- `/clear` — Clear conversation history
- `/status` — System status
- `/skills` — List skills
- `/capabilities` — Detailed skills and how to trigger them

Or ask **"What can you do?"** for a full capabilities list.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        WEB DASHBOARD (React)                        │
│   Chat │ Dashboard │ Setup │ Approvals │ Skills │ Settings │ Logs   │
│   Approvals: tool actions + self-healing code edits (diff preview)  │
└─────────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────────┐
│                         CORE ORCHESTRATOR                           │
│  Message Router │ Context Manager │ Security Guardian │ LLM Router  │
│  Memory System │ Skill Engine │ Self-Healing                        │
└─────────────────────────────────────────────────────────────────────┘
         │                    │                        │
┌────────────────┐  ┌─────────────────┐  ┌────────────────────────────┐
│ MESSAGING      │  │ STORAGE         │  │ EXECUTION                  │
│ Slack │ WA │ WS│  │ SQLite │ Chroma │  │ Docker Sandbox │ Direct    │
└────────────────┘  └─────────────────┘  └────────────────────────────┘
```

## Security Profiles

| Profile  | Read Files | Write Files | Shell | Messages | Web | Code Edit |
|----------|------------|-------------|-------|----------|-----|-----------|
| Paranoid | Approve    | Approve     | Approve| Approve | Approve | Approve |
| Balanced | Auto*      | Notify      | Approve| Approve | Auto | Approve |
| Trusted  | Auto       | Auto        | Notify | Notify  | Auto | Approve |

*Auto with path restrictions. **Code Edit**: Self-healing code fixes always require approval.

## Project Structure

```
aria/
├── config/                 # Configuration files
│   ├── settings.yaml      # Main config
│   └── self_healing.yaml  # Error patterns, pip packages, LLM fallback
├── docker/                 # Docker files
├── src/
│   ├── core/              # Orchestrator, LLM router, context
│   ├── security/          # Guardian, profiles, audit, sandbox
│   ├── memory/            # Short/long-term, episodic, RAG
│   ├── channels/          # Slack, WhatsApp, WebSocket
│   ├── skills/            # Built-in and learned skills
│   └── web/               # FastAPI backend + React frontend
├── whatsapp-bridge/       # Node.js WhatsApp bridge
└── tests/                 # Test suite
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/login` | POST | Login |
| `/api/chat/message` | POST | Send message |
| `/api/chat/history` | GET | Get chat history |
| `/api/approvals/pending` | GET | List pending approvals |
| `/api/approvals/respond` | POST | Respond to approval |
| `/api/skills` | GET | List skills |
| `/api/config` | GET | Get full config |
| `/api/config/llm` | PUT | Update LLM config |
| `/api/config/integrations` | PUT | Update integrations (Notion, Todoist, Linear, Spotify) |
| `/api/config/memory` | PUT | Update memory config |
| `/api/features` | GET | List all features |
| `/api/usage` | GET | LLM usage and cost stats |
| `/api/export` | GET | Export data (conversations, audit, full) |
| `/api/knowledge/process` | POST | Process knowledge graph |
| `/api/audit` | GET | Get audit log |
| `/api/docs` | GET | OpenAPI documentation |

## License

MIT
