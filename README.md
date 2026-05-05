# Personal WhatsApp Qdrant RAG Assistant

A personal WhatsApp assistant using **neonize** (WhatsApp client), **Ollama** (local LLM), **Qdrant** (vector search), and **MySQL** (structured data & logging).

## Architecture

```
WhatsApp → neonize → Message Router
                          ├── /admin  → Admin Commands → MySQL audit
                          ├── /remember, /forget → Memory Manager → Qdrant
                          └── (default) → RAG Engine → Qdrant + Ollama → Reply
```

## Quick Start

### 1. Configure Environment

```bash
cp .env.example .env
# Edit .env with your actual values
```

### 2. Run MySQL Migration

```bash
mysql -u rag -p whatsapp_rag < migrations/001_initial_schema.sql
```

### 3. Build & Run with Docker Compose

```bash
# On the server
docker compose build
docker compose up -d
docker logs -f personal-wa-assistant
```

### 4. Scan QR Code

On first run, the bot will display a QR code in the logs. Scan it with WhatsApp.

## Offline Build (Air-gapped Server)

```bash
# On a machine with internet, download wheels:
pip download -r requirements.txt -d wheelhouse/

# Copy wheelhouse/ to the server project directory
# The Dockerfile will use it automatically
```

## Admin Commands

| Command | Description |
|---|---|
| `/admin help` | Show available commands |
| `/admin listusers` | List all registered users |
| `/admin adduser <number> <name> <role>` | Add a user |
| `/admin setrole <number> <role>` | Change user role |
| `/admin blockuser <number>` | Block a user |
| `/admin unblockuser <number>` | Unblock a user |

## Memory Commands

| Command | Description |
|---|---|
| `/remember <fact>` | Save a fact to memory |
| `/memory search <query>` | Search your memories |
| `/forget <keyword>` | Delete a matching memory |

## Project Structure

```
source/
├── app/
│   ├── main.py              # Entry point
│   ├── config.py             # Settings from .env
│   ├── whatsapp_client.py    # Neonize wrapper
│   ├── message_router.py     # Message routing
│   ├── admin_commands.py     # /admin handler
│   ├── rag_engine.py         # RAG pipeline
│   ├── ollama_client.py      # Ollama HTTP client
│   ├── qdrant_store.py       # Qdrant manager
│   ├── mysql_store.py        # MySQL data layer
│   ├── memory_manager.py     # Memory commands
│   └── prompts.py            # Prompt templates
├── migrations/
│   └── 001_initial_schema.sql
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

## MySQL Tables

| Table | Purpose |
|---|---|
| `users` | Registered users with roles |
| `whatsapp_groups` | Tracked WhatsApp groups |
| `group_permissions` | Per-group bot settings |
| `documents` | Document metadata |
| `document_chunks` | Chunks with Qdrant point IDs |
| `conversation_logs` | Message history |
| `audit_logs` | Admin action trail |
| `failed_questions` | Unanswered questions |
| `inventory_items` | Personal inventory |

## Qdrant Collections

| Collection | Purpose |
|---|---|
| `personal_knowledge` | Documents, notes, guides |
| `personal_memory` | User-stored facts and preferences |
| `conversation_memory` | Summarized conversation context |

All collections: 1024-dim vectors, bge-m3 embeddings, Cosine distance.
