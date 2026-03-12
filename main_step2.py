import pandas as pd
import numpy as np
import re
from pathlib import Path
from rapidfuzz import process, fuzz

# --- কনফিগারেশন ---
BASE_DIR = Path.cwd()
DATA_RAW = BASE_DIR / "data_raw"
DATA_INT = BASE_DIR / "data_intermediate"
DATA_OUT = BASE_DIR / "data_output"

# ফোল্ডার তৈরি
for p in [DATA_INT, DATA_OUT]:
    p.mkdir(parents=True, exist_ok=True)

def normalize_name(s):
    s = str(s).lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# --- ধাপ ১: USDA ডাটা লোড ও ফিল্টার ---
print("🚀 Loading USDA Datasets...")
try:
    food = pd.read_csv(DATA_RAW / "food.csv", low_memory=False)
    nutrient = pd.read_csv(DATA_RAW / "nutrient.csv", low_memory=False)
    food_nutrient = pd.read_csv(DATA_RAW / "food_nutrient.csv", low_memory=False)

    # শুধুমাত্র প্রয়োজনীয় নিউট্রিয়েন্ট আইডি (Energy, Protein, Fat, Carbs, Fiber)
    # Energy: 1008, Protein: 1003, Fat: 1004, Carbs: 1005, Fiber: 1079 (USDA standard)
    target_nutrients = {1008: 'calories', 1003: 'protein', 1004: 'fat', 1005: 'carbs', 1079: 'fiber'}
    
    fn_filtered = food_nutrient[food_nutrient['nutrient_id'].isin(target_nutrients.keys())].copy()
    fn_filtered['nutrient_name'] = fn_filtered['nutrient_id'].map(target_nutrients)
    
    # Pivot করে এক লাইনে আনা
    macro_wide = fn_filtered.pivot_table(index="fdc_id", columns="nutrient_name", values="amount", aggfunc="mean").reset_index()
    
    # মূল ফুড লিস্টের সাথে যুক্ত করা
    master_fdc = food[['fdc_id', 'description', 'data_type']].merge(macro_wide, on='fdc_id', how='inner')
    master_fdc.rename(columns={'description': 'food_name'}, inplace=True)
    print(f"✅ USDA ডাটা প্রস্তুত: {master_fdc.shape[0]} টি খাবার পাওয়া গেছে।")
except Exception as e:
    print(f"❌ USDA ফাইল লোড করতে সমস্যা: {e}")

# --- ধাপ ২: ডেনসিটি (Density) ডাটা প্রসেস ---
print("\n⚖️ Processing Density Data for Volume-to-Weight conversion...")
try:
    dens_raw = pd.read_excel(DATA_RAW / "density_db.xlsx", sheet_name="Density DB")
    # কলাম ক্লিন করা
    dens_raw.columns = [str(c).strip().lower() for c in dens_raw.columns]
    
    # ডেনসিটি কলাম খুঁজে বের করা (সাধারণত ২য় কলামে থাকে)
    dens_df = dens_raw.iloc[:, [0, 1]].copy()
    dens_df.columns = ['food_name_density', 'density_g_per_ml']
    dens_df = dens_df.dropna()
    dens_df['density_name_norm'] = dens_df['food_name_density'].apply(normalize_name)
    print("✅ ডেনসিটি ডাটাবেজ প্রস্তুত।")
except Exception as e:
    print(f"❌ Density ফাইল লোড করতে সমস্যা: {e}")

# --- ধাপ ৩: ম্যানুয়াল ওভাররাইড (Common Foods) ---
# এই লিস্টটি VS Code-এ ডাটা ঠিক রাখতে খুব গুরুত্বপূর্ণ
manual_overrides = [
    {"keyword": "rice cooked", "unit": "cup", "grams": 200},
    {"keyword": "roti", "unit": "piece", "grams": 40},
    {"keyword": "egg whole", "unit": "piece", "grams": 50},
    {"keyword": "banana", "unit": "piece", "grams": 118},
    {"keyword": "milk", "unit": "cup", "grams": 250},
    {"keyword": "lentil dal", "unit": "cup", "grams": 200}
]

def apply_portions(row):
    name = normalize_name(row['food_name'])
    
    # ১. প্রথমে চেক করি ম্যানুয়াল ওভাররাইডে আছে কি না
    for ov in manual_overrides:
        if ov['keyword'] in name:
            return ov['grams'], ov['unit'], "manual"
    
    # ২. না থাকলে ডিফল্ট ১০০ গ্রাম (Solid food এর জন্য)
    return 100.0, "g", "default"

print("\n🛠️ Applying Portions and Weights...")
master_fdc[['grams_per_portion', 'portion_unit', 'method']] = master_fdc.apply(
    lambda r: pd.Series(apply_portions(r)), axis=1
)

# --- ধাপ ৪: সেভ করা ---
master_fdc.to_csv(DATA_OUT / "master_food_table_final.csv", index=False)
print(f"\n✅ সফলভাবে সেভ হয়েছে: {DATA_OUT / 'master_food_table_final.csv'}")