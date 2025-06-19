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
POSTGRES_USER=appuser
POSTGRES_PASSWORD=apppassword
POSTGRES_DB=appdb

TRITON__HOST=0.0.0.0
TRITON__PORT=8001

QDRANT__HOST=0.0.0.0
QDRANT__HTTP_PORT=6333
QDRANT__GRPC_PORT=6334
QDRANT__USE_SECURE=0
QDRANT__USE_GRPC=1
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

```python3
from grpclib.client import Channel
from tqdm import tqdm
from python_rag.proto.retriever import RetrieverStub
from python_rag.proto.text import TextDocumentAddRequest, TextRequest

channel = Channel("localhost", port=50081)
client = RetrieverStub(channel)

# Batch add
for i, x in tqdm(enumerate(processed)):
    if not x:
        continue
    await client.add_text_document(
        message=TextDocumentAddRequest(
            request_id=i,
            text=x,
            source="manual",
            tags=["manual"],
            models=["clip", "mpnet"],
        )
    )

# Search
res = await client.search_by_text(
    message=TextRequest(
        request_id=1,
        text="ancient country",
        source="manual",
        tags=["manual"],
        model="mpnet"
    )
)
```
> Protobuf definitions are located in `src/python_rag/proto/` and compiled automatically.

---

## 📈 Observability

Prometheus metrics are exposed for:
- Embedding latency
- Search performance
- Chunk insertion stats

---

## 🧱 Scalability Roadmap
 - Kubernetes manifests (Helm chart in progress)

 - FastAPI wrapper for browser clients and generation fusion

 - pytest tests

 - Auth layer for multi-tenant deployments


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