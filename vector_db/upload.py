from vector_db.settings import index
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
path_file = BASE_DIR / "data" / "datasets" / "ffatty" / "plain-text-wikipedia-simpleenglish" / "versions" / "2" / "AllCombined.txt"

model_emb = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")


def chunk_text(file, chunk_size=300):
  
  with open(path_file, "r", encoding="utf-8") as file:

    content = file.read()

  return [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]


chunks = chunk_text('AllCombined.txt', chunk_size=300)


wiki_emb = model_emb.encode(chunks, convert_to_tensor=True, show_progress_bar=True)

batch_size = 100
data_to_upsert = []

for i, (embedding, text) in enumerate(zip(wiki_emb, chunks)):
    data_to_upsert.append({
        "id": f"vec_{i}",
        "values": embedding.cpu().tolist(),
        "metadata": {"text": text}
    })

    # Upload when batch is full
    if len(data_to_upsert) == batch_size:
        index.upsert(vectors=data_to_upsert)
        print(f"Uploaded batch ending at vector {i}")

        # Clear the batch
        data_to_upsert.clear()

# Upload any remaining records
if data_to_upsert:
    index.upsert(vectors=data_to_upsert)
    print("Uploaded final partial batch")

print("Upload complete.")









