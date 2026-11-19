# scripts/01_prepare_data.py
import json
import os
import pandas as pd
from tqdm import tqdm

INPUT_FILE  = "arxiv-metadata-oai-snapshot.json"
OUTPUT_FILE = "data/arxiv_subset.parquet"
MAX_RECORDS = 10_000

os.makedirs("data", exist_ok=True)

def extract_year(paper: dict) -> int:
    try:
        versions = paper.get("versions", [])
        if versions:
            created = versions[0]["created"]  # "Mon, 2 Apr 2007 19:18:42 GMT"
            return int(created.split()[3])
    except (IndexError, ValueError, KeyError):
        pass
    return int(paper.get("update_date", "2000-01-01")[:4])

def format_authors(paper: dict) -> str:
    parsed = paper.get("authors_parsed", [])
    if parsed:
        parts = []
        for entry in parsed[:10]:
            last  = entry[0].strip() if len(entry) > 0 else ""
            first = entry[1].strip() if len(entry) > 1 else ""
            if last:
                parts.append(f"{last} {first}".strip())
        return ", ".join(parts)
    return paper.get("authors", "").replace("\n", " ")

records = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Читаємо датасет"):
        if len(records) >= MAX_RECORDS:
            break
        line = line.strip()
        if not line:
            continue
        paper = json.loads(line)

        abstract = paper.get("abstract", "").strip()
        title    = paper.get("title", "").strip()

        if not abstract or not title:
            continue

        categories_raw = paper.get("categories", "unknown")
        primary_category = categories_raw.split()[0]  

        records.append({
            "id":       paper["id"],
            "title":    title.replace("\n", " ").strip(),
            "abstract": abstract.replace("\n", " ").strip(),
            "authors":  format_authors(paper),
            "year":     extract_year(paper),
            "category": primary_category,
        })

# Варіант 2 (спеціально для виконання завдання з частини 3 можна було б одразу при завантаженні
# обмежити MAX_PER_EXAMPLE = 5000 і вибрати лише статті, які б задовільняли умови для Прикладу А та Прикладу Б)
# records_A = []  # Для Прикладу А 
# records_B = []  # Для Прикладу Б

#         # Перевірка для Прикладу А
#         if year >= 2021 and primary_category == "cs.LG":
#             if len(records_A) < MAX_PER_EXAMPLE:
#                 records_A.append({...})  # додаємо статтю
                
#         # Перевірка для Прикладу Б
#         elif year <= 2015:
#             if len(records_B) < MAX_PER_EXAMPLE:
#                 records_B.append({...})  # додаємо статтю

# І потім об'єднати обидва списки в один фінальний DataFrame на 10 000 записів

df = pd.DataFrame(records)
print(f"\nЗавантажено статей: {len(df)}")
print(f"\nРозподіл за категоріями (топ-10):")
print(df["category"].value_counts().head(10))
print(f"\nРозподіл за роками:")
print(df["year"].value_counts().sort_index().tail(10))
print(f"\nПриклад запису:")
print(df.iloc[0].to_dict())

df.to_parquet(OUTPUT_FILE, index=False)
print(f"\nЗбережено в {OUTPUT_FILE}")
