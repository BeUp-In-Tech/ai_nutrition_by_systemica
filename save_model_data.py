import pandas as pd
from pathlib import Path

# আপনার মেইন ফোল্ডার পাথ
BASE_DIR = Path.cwd()
input_path = BASE_DIR / "data_output" / "master_food_table_final.csv"
output_path = BASE_DIR / "data_output" / "final_smart_model.csv"

if input_path.exists():
    print("Reading large file... Please wait.")
    # শুধু প্রয়োজনীয় কলামগুলো লোড করছি যাতে মেমোরি কম লাগে
    cols_to_keep = ['food_name', 'calories', 'protein', 'carbs', 'fat', 'grams_per_portion']
    df = pd.read_csv(input_path, usecols=cols_to_keep)
    
    # ২০ ক্যালরির নিচের ফালতু আইটেমগুলো বাদ দিয়ে দিচ্ছি
    df = df[df['calories'] > 20].reset_index(drop=True)
    
    # ছোট ফাইল হিসেবে সেভ করছি
    df.to_csv(output_path, index=False)
    print(f"✅ Success! Your smart model data is saved at: {output_path}")
    print("Now your app will run 10x faster!")
else:
    print("❌ Error: Original file not found!")