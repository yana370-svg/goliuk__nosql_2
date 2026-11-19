# scripts/05_chunking.py
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Обов'язкове завантаження змінних оточення (.env) на початку файлу
load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768
CHUNK_SIZE_WORDS = 150
OVERLAP_WORDS = 20

# Ініціалізація бази Pinecone та моделі ембеддінгів
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")

# ============================================================================
# КРОК 1: ВИБІР 30 СТАТЕЙ З НАЙДОВШИМИ АНОТАЦІЯМИ
# ============================================================================
print("Аналіз довжини анотацій та вибір Топ-30...")
df["abstract_len"] = df["abstract"].apply(lambda x: len(str(x).split()))
top_30_df = df.nlargest(30, "abstract_len").reset_index(drop=True)

# ============================================================================
# КРОК 2: СТРАТЕГІЇ ЧАНКІНГУ (ФІКСОВАНА ТА СЕМАНТИЧНА)
# ============================================================================

def fixed_size_chunking(text, chunk_size=CHUNK_SIZE_WORDS, overlap=OVERLAP_WORDS):
    """Розбиття тексту на чанки фіксованого розміру з перекриттям слів."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) >= chunk_size // 3:  # Ігноруємо занадто короткі фінальні хвости
            chunks.append(chunk)
    return chunks

def semantic_chunking(text, max_words=CHUNK_SIZE_WORDS):
    """Об'єднання повних речень до досягнення ліміту слів для збереження змісту."""
    # Розділяємо текст на речення за крапками/знаками, зберігаючи структуру
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []
    current_len = 0
    
    for sent in sentences:
        sent_len = len(sent.split())
        if current_len + sent_len > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_len = sent_len
        else:
            current_chunk.append(sent)
            current_len += sent_len
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# ============================================================================
# КРОК 3: СТВОРЕННЯ ОКРЕНИХ ІНДЕКСІВ У PINECONE
# ============================================================================
print("Перевірка та створення векторних індексів у Pinecone...")
for idx_name in ["arxiv-chunks-fixed", "arxiv-chunks-semantic"]:
    if idx_name not in pc.list_indexes().names():
        print(f"Створення індексу: {idx_name}...")
        pc.create_index(
            name=idx_name, 
            dimension=VECTOR_DIM, 
            metric="cosine", 
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

index_fixed = pc.Index("arxiv-chunks-fixed")
index_semantic = pc.Index("arxiv-chunks-semantic")

# ============================================================================
# КРОК 4-5: ГЕНЕРАЦІЯ ЕМБЕДДІНГІВ ТА ЗАВАНТАЖЕННЯ ЧАНКІВ БАТЧАМИ
# ============================================================================
print("\nОбробка, кодування та завантаження чанків на сервер...")
for _, row in tqdm(top_30_df.iterrows(), total=len(top_30_df), desc="Обробка статей"):
    text = row["abstract"]
    arxiv_id = row["id"]
    title = row["title"]
    
    # --- Стратегія Fixed-size ---
    fixed_chunks = fixed_size_chunking(text)
    vectors_fixed = []
    for i, chunk in enumerate(fixed_chunks):
        # ВАЖЛИВО: Обов'язкова склейка з [SEP] для моделі SPECTER 2
        input_text = f"{title} [SEP] {chunk}"
        emb = model.encode(input_text, normalize_embeddings=True).tolist()
        
        vectors_fixed.append({
            "id": f"{arxiv_id}_fixed_{i}",
            "values": emb,
            "metadata": {
                "arxiv_id": arxiv_id, 
                "title": title[:200], 
                "chunk_text": chunk,  # Зберігаємо весь текст чанка без обрізки
                "chunk_idx": i, 
                "year": int(row["year"]), 
                "category": str(row["category"])
            }
        })
    if vectors_fixed: 
        index_fixed.upsert(vectors=vectors_fixed)

    # --- Стратегія Semantic ---
    semantic_chunks = semantic_chunking(text)
    vectors_semantic = []
    for i, chunk in enumerate(semantic_chunks):
        # ВАЖЛИВО: Обов'язкова склейка з [SEP] для моделі SPECTER 2
        input_text = f"{title} [SEP] {chunk}"
        emb = model.encode(input_text, normalize_embeddings=True).tolist()
        
        vectors_semantic.append({
            "id": f"{arxiv_id}_semantic_{i}",
            "values": emb,
            "metadata": {
                "arxiv_id": arxiv_id, 
                "title": title[:200], 
                "chunk_text": chunk,  # Зберігаємо весь текст чанка без обрізки
                "chunk_idx": i, 
                "year": int(row["year"]), 
                "category": str(row["category"])
            }
        })
    if vectors_semantic: 
        index_semantic.upsert(vectors=vectors_semantic)

print("Завантаження всіх типів чанків успішно завершено.")

# ============================================================================
# КРОК 6: РЕАЛІЗАЦІЯ ФУНКЦІЇ ПОШУКУ ПО ЧАНКАХ
# ============================================================================
def search_chunks(query, index_obj, top_k=3, strategy_label=""):
    """Виконує семантичний пошук фрагментів за вектором запиту."""
    # Для пошуку науковою моделлю додаємо базову структуру
    q_vec = model.encode(query, normalize_embeddings=True).tolist()
    res = index_obj.query(vector=q_vec, top_k=top_k, include_metadata=True)
    
    print(f"\nРЕЗУЛЬТАТІ ПОШУКУ ДЛЯ СТРАТЕГІЇ [{strategy_label}] ЗА ЗАПИТОМ: '{query}':")
    print("-" * 80)
    for m in res['matches']:
        meta = m['metadata']
        print(f"  Фрагмент №{meta['chunk_idx']} | Схожість (Score): {m['score']:.4f}")
        print(f"  Стаття: {meta['title'][:65]}...")
        print(f"  Текст чанка: {meta['chunk_text'][:150]}...\n")

# Запуск декількох тестових пошуків для порівняння стратегій
search_chunks("experimental results show significant improvement", index_fixed, strategy_label="FIXED-SIZE")
search_chunks("experimental results show significant improvement", index_semantic, strategy_label="SEMANTIC")

search_chunks("quantum chromodynamics perturbative calculation", index_fixed, strategy_label="FIXED-SIZE")
search_chunks("quantum chromodynamics perturbative calculation", index_semantic, strategy_label="SEMANTIC")
