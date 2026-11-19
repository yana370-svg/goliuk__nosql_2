# scripts/03_load_to_pinecone.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Переконуємося, що .env завантажується першим
load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200

# Ініціалізація клієнта
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# 1. Створення індексу, якщо не існує
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Створення індексу {INDEX_NAME}...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
else:
    print(f"Індекс {INDEX_NAME} вже існує.")

# Підключення до індексу та завантаження локальних файлів
index = pc.Index(INDEX_NAME)
df = pd.read_parquet(INPUT_PARQUET)
embeddings = np.load(INPUT_EMBEDDINGS)

print("Початок підготовки та завантаження даних батчами...")

# Виправлено: ітеруємося відразу кроками по BATCH_SIZE (по 200 елементів)
for start_idx in tqdm(range(0, len(df), BATCH_SIZE), desc="Завантаження в Pinecone"):
    end_idx = start_idx + BATCH_SIZE
    
    # Вирізаємо поточну пачку даних
    chunk_df = df.iloc[start_idx:end_idx]
    chunk_vectors = embeddings[start_idx:end_idx]
    
    vectors_to_upsert = []
    
    # Формуємо об'єкти всередині батчу
    for idx, (local_idx, row) in enumerate(chunk_df.iterrows()):
        # Розраховуємо глобальний порядковий номер для ID вигляду paper_<номер>
        global_num = start_idx + idx
        
        meta = {
            "arxiv_id": str(row["id"]),
            "title": str(row["title"])[:300],
            "abstract": str(row["abstract"])[:500],
            "authors": str(row["authors"])[:200],
            "year": int(row["year"]),
            "category": str(row["category"])
        }
        
        vectors_to_upsert.append({
            "id": f"paper_{global_num}",
            "values": chunk_vectors[idx].tolist(),
            "metadata": meta
        })
        
    # Надсилаємо сформовану пачку з 200 елементів на сервер
    index.upsert(vectors=vectors_to_upsert)

# 5. Вивід загальної кількості векторів
stats = index.describe_index_stats()
print(f"\nЗавантаження завершено успішно!")
print(f"Загальна кількість векторів в індексі: {stats['total_vector_count']}")
