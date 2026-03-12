import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD
from pathlib import Path

# --- কনফিগারেশন ও পাথ ---
BASE_DIR = Path.cwd()
DATA_OUT = BASE_DIR / "data_output"
master_food_path = DATA_OUT / "master_food_table_final.csv"

# ডাটা লোড ও ক্লিনিং
df_foods = pd.read_csv(master_food_path)
cols = ['calories', 'protein', 'carbs', 'fat', 'grams_per_portion']
df_foods[cols] = df_foods[cols].fillna(0)
df_foods = df_foods[df_foods['calories'] > 20].reset_index(drop=True) # খুব কম ক্যালরির খাবার বাদ

# --- ১. উন্নত ইউজার প্রোফাইল ক্যালকুলেটর ---
def get_user_profile(age, gender, weight, height, activity, goal):
    # BMR calculation
    if gender.lower() == 'male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    
    factors = {"sedentary": 1.2, "lightly": 1.375, "moderate": 1.55, "very": 1.725}
    tdee = bmr * factors.get(activity, 1.55)
    
    # Goal logic
    target_cal = tdee - 500 if goal == "weight_loss" else (tdee + 300 if goal == "muscle_gain" else tdee)
    
    return {
        "bmi": round(weight / ((height/100)**2), 2),
        "bmr": round(bmr, 1),
        "tdee": round(tdee, 1),
        "targets": {
            "calories": round(target_cal, 1),
            "protein": round((target_cal * 0.25) / 4, 1),
            "carbs": round((target_cal * 0.45) / 4, 1),
            "fat": round((target_cal * 0.30) / 9, 1)
        }
    }

# --- ২. মিল অপ্টিমাইজার ফাংশন ---
def solve_meal(food_pool, target_cal, target_prot):
    prob = LpProblem("Meal_Sub", LpMinimize)
    food_vars = LpVariable.dicts("F", food_pool.index, 0, 2) # সর্বোচ্চ ২ পোরশন
    prob += 1 # Dummy objective
    
    # Constraints
    prob += lpSum([food_vars[i] * food_pool.loc[i, 'calories'] * (food_pool.loc[i, 'grams_per_portion']/100) for i in food_pool.index]) >= target_cal * 0.9
    prob += lpSum([food_vars[i] * food_pool.loc[i, 'calories'] * (food_pool.loc[i, 'grams_per_portion']/100) for i in food_pool.index]) <= target_cal * 1.1
    prob += lpSum([food_vars[i] * food_pool.loc[i, 'protein'] * (food_pool.loc[i, 'grams_per_portion']/100) for i in food_pool.index]) >= target_prot * 0.8
    
    prob.solve(PULP_CBC_CMD(msg=0))
    
    results = []
    if prob.status == 1:
        for i in food_pool.index:
            if food_vars[i].varValue and food_vars[i].varValue > 0.1:
                v = food_vars[i].varValue
                g = food_pool.loc[i, 'grams_per_portion']
                results.append({
                    "name": food_pool.loc[i, 'food_name'][:40],
                    "grams": int(v * g),
                    "kcal": int(v * food_pool.loc[i, 'calories'] * (g/100)),
                    "p": round(v * food_pool.loc[i, 'protein'] * (g/100), 1),
                    "c": round(v * food_pool.loc[i, 'carbs'] * (g/100), 1),
                    "f": round(v * food_pool.loc[i, 'fat'] * (g/100), 1)
                })
    return results

# --- ৩. মেইন আউটপুট ডিসপ্লে ---
profile = get_user_profile(age=25, gender="male", weight=80, height=175, activity="moderate", goal="weight_loss")
targets = profile['targets']

print(f"\n👤 Profile:\n   BMI={profile['bmi']}  BMR={profile['bmr']}  TDEE={profile['tdee']}")
print(f"   Targets → {targets}\n")
print("📋 Building daily plan...\n")

meals = [
    ("🌅 BREAKFAST", 0.25), ("🍎 SNACK 1", 0.10), ("🍛 LUNCH", 0.30), 
    ("☕ SNACK 2", 0.10), ("🌙 DINNER", 0.25)
]

print("="*68)
print("   🥗  OPTIMIZED DAILY MEAL PLAN")
print("="*68)

total_actual = {'kcal':0, 'p':0, 'c':0, 'f':0}

for name, ratio in meals:
    m_cal = targets['calories'] * ratio
    m_prot = targets['protein'] * ratio
    
    # প্রতি মিলের জন্য আলাদা খাবারের পুল (রেন্ডমাইজড)
    sample = df_foods.sample(200).reset_index(drop=True)
    res = solve_meal(sample, m_cal, m_prot)
    
    print(f"\n  {name}  (target {int(m_cal)} kcal)")
    print("  " + "─"*64)
    
    m_kcal = 0
    for f in res:
        print(f"     • {f['name']:<40} {f['grams']:>4}g    {f['kcal']:>3} kcal")
        print(f"                                                P {f['p']} C {f['c']} F {f['f']}")
        m_kcal += f['kcal']
        total_actual['kcal'] += f['kcal']; total_actual['p'] += f['p']
        total_actual['c'] += f['c']; total_actual['f'] += f['f']
    
    print(f"     total →                                    {m_kcal} kcal")

print("\n" + "="*68)
print("   📊  TOTALS  vs  TARGETS")
print("="*68)
print(f"                     Target    Actual   Accuracy")
print("   " + "─"*14 + " " + "─"*9 + " " + "─"*9 + " " + "─"*10)

for label, key, t_val in [("Calories", "kcal", targets['calories']), ("Protein g", "p", targets['protein']), 
                          ("Carbs g", "c", targets['carbs']), ("Fat g", "f", targets['fat'])]:
    acc = min(100.0, (total_actual[key]/t_val)*100)
    print(f"   {label:<14} {t_val:>9} {round(total_actual[key],1):>9}    {round(acc,1)}%  ✅")

print(f"\n   OVERALL...............................     {round((total_actual['kcal']/targets['calories'])*100, 1)}%")
print("="*68)