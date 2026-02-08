# Aria - Personal AI Assistant

A fully-featured personal AI assistant that runs locally with hybrid LLM support (local Ollama + cloud Claude), multi-channel messaging (Slack, WhatsApp, Web), configurable security profiles, and extensible skills.

## Features

- **Hybrid LLM**: Routes simple tasks to local Ollama, complex reasoning to Claude API
- **Smart Ollama Setup**: Hardware-detection during onboarding suggests the best model for your machine; can download it for you
- **Multi-Channel**: Slack, WhatsApp (via web bridge), and Web UI
- **Security**: Configurable profiles (paranoid, balanced, trusted) with approval workflows
- **Sandboxed Execution**: Docker isolation for untrusted code
- **Memory**: Short-term conversation, long-term vector store (ChromaDB), episodic task history, user profiles
- **Knowledge Graph**: Cognee integration for entity/relationship extraction—synced with user profiles
- **29 Features**: Morning briefing, context reminders, proactive suggestions, time-of-day awareness, cost/usage tracking, theme toggle, data export, PWA, push notifications, keyboard shortcuts, debug trace, skill templates, and more
- **Integrations**: Notion, Todoist, Linear, Spotify (API keys configurable in Settings or onboarding)
- **Built-in Skills**: File ops, shell, browser, calendar, email, SMS, TTS, STT, image, video, documents, weather, research, agent, memory, finance, news, contacts, tracking, webhook, home, notion, todoist, linear, spotify
- **Dynamic Skills**: AI can generate new skills on demand
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

### Running

**Development mode**:
```bash
# Terminal 1: Start backend
python -m src.main

# Terminal 2: Start frontend dev server
cd src/web/frontend
npm run dev
```

**With Docker**:
```bash
cd docker
docker-compose up -d
```

### Configuration

Edit `config/settings.yaml` to customize:
- LLM providers and models
- Channel settings (Slack, WhatsApp, Web)
- Security profile
- Memory, knowledge graph (Cognee), and sandbox options
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
│   Chat │ Dashboard │ Setup │ Approvals │ Skills │ Settings │ Logs  │
└─────────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────────┐
│                         CORE ORCHESTRATOR                           │
│  Message Router │ Context Manager │ Security Guardian │ LLM Router  │
│  Memory System │ Skill Engine                                       │
└─────────────────────────────────────────────────────────────────────┘
         │                    │                        │
┌────────────────┐  ┌─────────────────┐  ┌────────────────────────────┐
│ MESSAGING      │  │ STORAGE         │  │ EXECUTION                  │
│ Slack │ WA │ WS│  │ SQLite │ Chroma │  │ Docker Sandbox │ Direct    │
└────────────────┘  └─────────────────┘  └────────────────────────────┘
```

## Security Profiles

| Profile  | Read Files | Write Files | Shell | Messages | Web |
|----------|------------|-------------|-------|----------|-----|
| Paranoid | Approve    | Approve     | Approve| Approve | Approve |
| Balanced | Auto*      | Notify      | Approve| Approve | Auto |
| Trusted  | Auto       | Auto        | Notify | Notify  | Auto |

*Auto with path restrictions

## Project Structure

```
aria/
├── config/                 # Configuration files
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
