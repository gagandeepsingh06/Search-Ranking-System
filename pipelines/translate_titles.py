"""
translate_titles.py
-------------------
Translates Hindi product titles to English ONCE
and saves them back to the CSV as a new column 'title_en'.

Run ONCE with:
    python -m pipelines.translate_titles
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from deep_translator import GoogleTranslator
from configs.settings import DATA_PATH


def translate_text(text):
    try:
        return GoogleTranslator(source="hi", target="en").translate(str(text))
    except:
        return text


print("\n" + "="*50)
print("  TRANSLATING TITLES")
print("="*50)

df = pd.read_csv(DATA_PATH)
print(f"\nLoaded {len(df)} rows")

# Skip if already translated
if "title_en" in df.columns:
    print("title_en column already exists — skipping.")
else:
    print("Translating titles... (this runs only once)\n")
    translated = []
    for i, title in enumerate(df["title"]):
        translated.append(translate_text(title))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(df)} done...")

    df["title_en"] = translated
    df.to_csv(DATA_PATH, index=False)
    print(f"\nSaved translated titles to: {DATA_PATH}")

print("\n" + "="*50)
print("  DONE")
print("="*50 + "\n")