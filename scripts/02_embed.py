import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv
import torch

INPUT_PARQUET = "data/arxiv_subset.parquet"
OUTPUT_DIR = "embeddings"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "embeddings.npy")
MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 64

def main():
    # Сначала загружаем переменные окружения из .env
    load_dotenv()
    # Завантаження даних
    # torch.set_num_threads(8) 
    print("Завантаження датасету...")
    df = pd.read_parquet(INPUT_PARQUET)
    
    # 2. Підготовка текстів: title + " [SEP] " + abstract
    print("Підготовка текстів...")
    texts = (df["title"] + " [SEP] " + df["abstract"]).tolist()
    
    # 3. Завантаження моделі
    print(f"Завантаження моделі {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    # 4. Генерація ембеддингів
    print("Кодування текстів у вектори...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True  # Критично важливо для косинусної схожості
    )
    
    # 5. Вивід інформації
    print(f"\nЗагальна кількість оброблених текстів: {len(embeddings)}")
    print(f"Розмірність ембеддингів: {embeddings.shape[1]}")
    print(f"Норма першого ембеддингу: {np.linalg.norm(embeddings[0]):.6f} (має бути ~1.0)")
    
    # 6. Збереження
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(OUTPUT_FILE, embeddings)
    print(f"Ембеддинги збережено у {OUTPUT_FILE}")

if __name__ == "__main__":
    main()