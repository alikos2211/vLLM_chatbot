# RAG Chatbot with GPT-2 + Pinecone + vLLM

A retrieval-augmented generation (RAG) chatbot that serves GPT-2 via a local OpenAI-compatible endpoint, backed by a Pinecone vector store of Simple English Wikipedia.

---

## Different Workflows

The project supports two different model serving workflows:

---

## 1. vLLM + Docker Workflow

This workflow uses **vLLM** inside a Docker container for optimized inference.

### Features

- High-performance inference engine
- Continuous batching for multiple requests
- OpenAI-compatible API server
- Containerized deployment

### Flow

1. GPT-2 model loaded into vLLM
2. Docker container starts API server
3. OpenAI-compatible endpoint exposed
4. RAG pipeline injects retrieved context
5. vLLM generates response

---

## 2. Hugging Face Transformers Workflow

This workflow uses the Hugging Face `transformers` library for direct model loading.

### Features

- Simple Python-based setup
- Easy debugging and customization
- Lightweight deployment
- Lower performance compared to vLLM

### Flow

1. GPT-2 loaded via `transformers`
2. RAG pipeline adds retrieved context
3. Model generates response

## Workflow Comparison

| Feature | vLLM + Docker | Hugging Face Transformers |
|--------|---------------|----------------------------|
| Performance | High | Medium |
| Scalability | High concurrency | Limited concurrency |
| Setup Complexity | Medium–High | Low |
| Deployment | Production-ready | Development-friendly |
| API Compatibility | Native OpenAI-compatible | Custom implementation |
| Resource Efficiency | Optimized GPU usage | Standard PyTorch usage |


## Architecture

```
User input → Encode (sentence-transformers) → Pinecone top-5 → Build prompt → GPT-2 via vLLM → Response
```

| Component | Role |
|---|---|
| **vLLM** | Serves GPT-2 on a local OpenAI-compatible endpoint at `http://localhost:8000/v1/` |
| **Pinecone (serverless)** | Vector store of 300-character Wikipedia chunks as dense embeddings |
| **sentence-transformers** | Encodes queries and documents |
| **OpenAI Python SDK** | Client for the vLLM `/v1` endpoint |

---

## How It Works

### Data Ingestion
The Simple English Wikipedia dataset is downloaded via `kagglehub`, split into 300-character chunks, encoded with `sentence-transformers`, and uploaded to a Pinecone serverless vector index in batches of 100.

### Retrieval
At query time, the user's message is encoded and the top-5 nearest chunks are fetched from Pinecone.

### Generation
Retrieved context is prepended to a rolling conversation prompt; GPT-2 completes it via the vLLM `/v1` API.

### Conversation Management
A sliding window of the last 6 turns is kept in memory to provide context without exceeding GPT-2's 1024-token limit. The CLI supports the following commands:

| Command | Description |
|---|---|
| `quit` | Exit the chatbot |
| `clear` | Clear conversation history |
| `history` | Show conversation history |


---

## Tech Stack

- [GPT-2](https://github.com/openai/gpt-2 )
- [vLLM](https://github.com/vllm-project/vllm)
- [Pinecone (serverless)](https://www.pinecone.io/)
- [sentence-transformers](https://www.sbert.net/)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [kagglehub](https://github.com/Kaggle/kagglehub)
- [Simple English Wikipedia](https://www.kaggle.com/datasets/ffatty/plain-text-wikipedia-simpleenglish)

---

## Speed Optimization Techniques, which can be used in future

Listed by impact-to-effort ratio:

### 1. GPU Acceleration *(highest impact)*
Moving vLLM from CPU to any modern GPU is by far the biggest win. vLLM uses PagedAttention and continuous batching — generation goes from ~20 tok/s (CPU) to **200–300+ tok/s** on a T4. Works on free-tier Google Colab.

### 2. Quantization
Load GPT-2 in 8-bit or 4-bit precision using `bitsandbytes`. This halves or quarters memory usage and speeds up generation at minimal quality cost.

```python
args = parser.parse_args([
    "--model", model,
    "--quantization", "awq",   # or "gptq", "bitsandbytes"
    "--dtype", "float16",
    ...
])
```

### 3. Reduce `max_tokens`
The current limit is 100 tokens. Reducing to 60–70 cuts generation time proportionally if shorter answers are acceptable.

### 4. Limit Context Window
`--max-model-len 1024` is GPT-2's hard limit. Keep the prompt tight — retrieved chunks at 300 characters each add up fast. Consider reducing `top_k` from 5 to 3 if retrieval quality allows.

### 5. Async Embedding + Retrieval
Overlap the Pinecone network call with pre-building fixed parts of the prompt:

```python
import asyncio

async def retrieve(query_vector):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: index.query(...))
```

### 6. Cache Frequent Queries
Cache `(query_vector, retrieved_chunks)` pairs with an LRU cache or Redis for repeated queries on the same topics. The Pinecone round-trip is free to skip when the result is already cached.

### 7. Use a Faster Embedding Model
`sentence-transformers/all-MiniLM-L6-v2` is smaller and faster than heavier models, and works well for retrieval. Switching to MiniLM can cut embedding time by **3–5×**.

---

# vLLM Setup via Docker on Windows

A step-by-step guide to running a vLLM OpenAI-compatible server on Windows using Docker Desktop with NVIDIA GPU support.

---


## Step 1 — Install Docker Desktop

1. Download Docker Desktop from: 👉 https://www.docker.com/products/docker-desktop/
2. Run the installer.
3. After installation, open Docker Desktop.


---

##  Step 2 — Run vLLM with Docker

1. Build an Image. Run comand:
```bash
  docker build -t vllm_image:latest .
```
2. Build Container. Run comand:
```bash
  docker run --gpus all -p 8000:8000 vllm_image
```
---

## Step 5 — Verify the Server

Check the server is running and the model is loaded:

```bash
  curl http://localhost:8000/v1/models
```

Expected response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt2",
      "object": "model",
      ...
    }
  ]
}
```

---

## Installation of pacages  and Run

1. Install Python v3.11.15
2. Install Pytorch CUDA 12.8. Run the comand:
```bash
  pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```
3. Install the Requirement list. Run the comand:
```bash
  pip install -r requirements.py
```
4. Run the run.py file. Run the command:
```bash
  python run.py
```

## 📚 Useful Links

- [vLLM Documentation](https://docs.vllm.ai)
- [vLLM Docker Hub](https://hub.docker.com/r/vllm/vllm-openai)


