# Python-RAG: GRPC-based Embedding Retrieval Engine

This project implements a scalable, modular retrieval system using the following components:

- **NVIDIA Triton Inference Server** for embedding generation  
- **Qdrant** as a high-performance vector store  
- **gRPC** as the communication layer  
- **Configuration** via `.env` or environment variables  
- **Async Python** stack with full Prometheus observability

---

## 🔧 Project Structure

```
python-rag/
├── src/
│   ├── python_rag/
│   │   ├── server/              # gRPC server entrypoint
│   │   ├── proto/               # Compiled and raw proto definitions
│   │   ├── qdrant/              # Qdrant vector DB integration
│   │   ├── triton/              # Triton embedding client
│   │   ├── dto/                 # Data models (Chunks, Embeddings, etc.)
│   │   ├── config/              # Configuration logic (env, defaults, validation)
│   │   ├── monitoring/          # Prometheus metrics
│   │   └── ...                  # Additional logic
├── docker-compose.yml          # Local deployment: Qdrant + Triton
├── .env.example                # Example environment configuration
└── README.md                   # You are here
```

---

## ⚙️ Components

### 1. Triton Inference Server
Used to compute dense vector embeddings from input text or image. You can plug in your own model by updating the model repository path in the `.env`.

### 2. Qdrant Vector Store
Used to store and retrieve embeddings using approximate nearest neighbor (ANN) search with HNSW and optional Product Quantization (PQ).

### 3. gRPC Interface
The system is exposed **entirely via gRPC** for high-throughput, low-latency access. Clients can:
- Insert documents or chunks
- Search by embedding
- Remove entries by document or chunk ID

### 4. Configuration
Configurable via:
- `.env` file
- Or directly via environment variables

Example `.env` file:
```dotenv
TRITON_HTTP_PORT=8000
TRITON_GRPC_PORT=8001
TRITON_HOST=triton
TRITON_MODEL_NAME=mpnet

QDRANT_HOST=qdrant
QDRANT_HTTP_PORT=6333
QDRANT_GRPC_PORT=6334
QDRANT_USE_GRPC=true
QDRANT_USE_SECURE=false
QDRANT_TIMEOUT=10.0

COLLECTION_NAME=clip-image
VECTOR_SIZE=768
DISTANCE=cosine
```

---

## 🚀 Getting Started

### Step 1. Clone the repository

```bash
git clone https://github.com/kirillemilio/python-rag.git
cd python-rag
```

### Step 2. Setup environment

```bash
cp .env.example .env
# Then modify values as needed
```

### Step 3. Start services with Docker Compose

```bash
docker-compose up --build
```

This will start:
- `triton` on ports 8000/8001
- `qdrant` on 6333/6334

### Step 4. Run the gRPC server

```bash
python -m python_rag.server.server
```

This will initialize:
- Triton client
- Qdrant connection
- Collection creation if it doesn’t exist
- Start accepting gRPC requests

---

## 📡 gRPC API

All functionality is exposed via gRPC.

### Example Methods

- `SearchByText`: Generate embedding via Triton → search in Qdrant  
- `InsertChunk`: Add a new embedding vector to collection  
- `DeleteById`: Remove a vector from Qdrant  
- `ClearCollection`: Drop all vectors

> Protobuf definitions are located in `src/python_rag/proto/` and compiled automatically.

---

## 📈 Observability

Prometheus metrics are exposed for:
- Embedding latency
- Search performance
- Chunk insertion stats

---

## 🧪 Testing

To test functionality manually or via CI:

```bash
pytest
```

Or use a gRPC client like `grpcurl` or `grpcui`.

---

## 🧠 Notes

- **Batch Insertion:** Use `add_chunks_in_batches()` to optimize write throughput to Qdrant.
- **Security:** Currently no TLS. Production deployments should secure traffic.
- **Scalability:** Easily containerizable and horizontally scalable with message queues or FastAPI wrappers.

---

## 📃 License

MIT License. Use at your own risk. Contributions welcome.

---

## 👤 Author

Made by @kirillemilio — a lightweight async vector search engine with gRPC ✨.