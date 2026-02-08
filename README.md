# Aria - Personal AI Assistant

A fully-featured personal AI assistant that runs locally with hybrid LLM support (local Ollama + cloud Claude), multi-channel messaging (Slack, WhatsApp, Web), configurable security profiles, and extensible skills.

## Features

- **Hybrid LLM**: Routes simple tasks to local Ollama, complex reasoning to Claude API
- **Multi-Channel**: Slack, WhatsApp (via web bridge), and Web UI
- **Security**: Configurable profiles (paranoid, balanced, trusted) with approval workflows
- **Sandboxed Execution**: Docker isolation for untrusted code
- **Memory**: Short-term conversation, long-term vector store (ChromaDB), episodic task history
- **11 Built-in Skills**: File ops, shell, browser, calendar, email, SMS, TTS, STT, image, video, documents
- **Dynamic Skills**: AI can generate new skills on demand
- **Web Dashboard**: Full React UI for chat, approvals, settings, and logs

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

2. **Set up environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Install frontend dependencies**:
   ```bash
   cd src/web/frontend
   npm install
   ```

4. **(Optional) Set up WhatsApp bridge**:
   ```bash
   cd whatsapp-bridge
   npm install
   ```

5. **(Optional) Pull Ollama models**:
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
- Memory and sandbox options

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        WEB DASHBOARD (React)                        │
│   Chat │ Approvals │ Settings │ Logs │ Skills                       │
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
| `/api/chat` | POST | Send message |
| `/api/chat/history` | GET | Get chat history |
| `/api/approvals/pending` | GET | List pending approvals |
| `/api/approvals/{id}` | POST | Respond to approval |
| `/api/skills` | GET | List skills |
| `/api/settings` | GET/PUT | Get/update settings |
| `/api/audit` | GET | Get audit log |

## License

MIT
