from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD
from pathlib import Path
from typing import List, Dict, Optional

app = FastAPI(title="EatWise Pro API - Medical Grade")

# --- Data Setup ---
BASE_DIR = Path.cwd()
data_path = BASE_DIR / "data_output" / "final_smart_model.csv"

if data_path.exists():
    df_foods = pd.read_csv(data_path)
    cols = ['calories', 'protein', 'carbs', 'fat', 'grams_per_portion']
    df_foods[cols] = df_foods[cols].fillna(0)
    # Filter items with > 20 calories to avoid non-meal items
    df_foods = df_foods[df_foods['calories'] > 20].reset_index(drop=True)
else:
    print("❌ Error: master_food_table_final.csv not found!")

# --- Input Model ---
class UserProfile(BaseModel):
    age: int
    gender: str
    weight: float
    height: float
    activity: str # sedentary, lightly_active, moderately_active, very_active, extra_active
    goal: str     # extreme_weight_loss, weight_loss, maintain, muscle_gain, extreme_muscle_gain
    has_diabetes: bool = False
    allergies: List[str] = []

# --- Logic Functions ---

def get_targets(u: UserProfile):
    # BMR Calculation (Mifflin-St Jeor Equation)
    if u.gender.lower() == 'male':
        bmr = 10 * u.weight + 6.25 * u.height - 5 * u.age + 5
    else:
        bmr = 10 * u.weight + 6.25 * u.height - 5 * u.age - 161
    
    activity_factors = {
        "sedentary": 1.2, "lightly_active": 1.375, 
        "moderately_active": 1.55, "very_active": 1.725, "extra_active": 1.9
    }
    tdee = bmr * activity_factors.get(u.activity.lower(), 1.55)
    
    goal_adjustments = {
        "extreme_weight_loss": -1000, "weight_loss": -500, 
        "maintain": 0, "muscle_gain": 400, "extreme_muscle_gain": 800
    }
    target_cal = tdee + goal_adjustments.get(u.goal.lower(), 0)
    
    return {
        "calories": round(target_cal, 1),
        "protein": round((target_cal * 0.25) / 4, 1),
        "carbs": round((target_cal * 0.45) / 4, 1),
        "fat": round((target_cal * 0.30) / 9, 1)
    }

def get_personalized_tips(u: UserProfile, targets: Dict):
    bmi = round(u.weight / ((u.height/100)**2), 2)
    tips = []
    
    if bmi < 18.5:
        tips.append("Your BMI indicates you are underweight. Focus on calorie-dense healthy foods.")
    elif bmi > 25:
        tips.append("Your BMI is high. Prioritize fiber and lean protein to stay full.")
    
    if u.has_diabetes:
        tips.append("Diabetes Care: Your plan excludes high-sugar items. Stick to complex carbs.")
    
    if u.allergies:
        tips.append(f"Allergy Alert: Items containing {', '.join(u.allergies)} have been filtered out.")

    tips.append("Ensure 7-8 hours of sleep for better metabolic health.")
    return tips

def solve_day_plan(targets, u: UserProfile):
    # Medical Filtering
    pool = df_foods.copy()
    
    for allergy in u.allergies:
        pool = pool[~pool['food_name'].str.contains(allergy, case=False, na=False)]
    
    if u.has_diabetes:
        forbidden = ['sugar', 'syrup', 'candy', 'soda', 'sweetened', 'dessert']
        for word in forbidden:
            pool = pool[~pool['food_name'].str.contains(word, case=False, na=False)]

    meals_config = [
        ("Breakfast", 0.20), ("Snack 1", 0.15), ("Lunch", 0.30), 
        ("Snack 2", 0.10), ("Dinner", 0.25)
    ]
    
    day_plan = {}
    for name, ratio in meals_config:
        m_cal = targets['calories'] * ratio
        sample = pool.sample(min(300, len(pool))).reset_index(drop=True)
        
        prob = LpProblem(f"Meal_{name}", LpMinimize)
        food_vars = LpVariable.dicts("F", sample.index, 0, 2)
        prob += 1
        prob += lpSum([food_vars[i] * sample.loc[i, 'calories'] * (sample.loc[i, 'grams_per_portion']/100) for i in sample.index]) >= m_cal * 0.95
        prob += lpSum([food_vars[i] * sample.loc[i, 'calories'] * (sample.loc[i, 'grams_per_portion']/100) for i in sample.index]) <= m_cal * 1.05
        
        prob.solve(PULP_CBC_CMD(msg=0))
        
        meal_items = []
        if prob.status == 1:
            for i in sample.index:
                if food_vars[i].varValue and food_vars[i].varValue > 0.1:
                    meal_items.append({
                        "item": sample.loc[i, 'food_name'],
                        "amount": f"{int(food_vars[i].varValue * sample.loc[i, 'grams_per_portion'])}g",
                        "kcal": int(food_vars[i].varValue * sample.loc[i, 'calories'] * (sample.loc[i, 'grams_per_portion']/100))
                    })
        day_plan[name] = meal_items
    return day_plan

# --- API ENDPOINTS ---

@app.post("/daily-plan")
async def daily_plan(user: UserProfile):
    targets = get_targets(user)
    plan = solve_day_plan(targets, user)
    tips = get_personalized_tips(user, targets)
    return {
        "type": "Daily Plan",
        "user_stats": {"bmi": round(user.weight / ((user.height/100)**2), 2), "targets": targets},
        "plan": plan,
        "health_tips": tips
    }

@app.post("/weekly-plan")
async def weekly_plan(user: UserProfile):
    targets = get_targets(user)
    weekly_data = {day: solve_day_plan(targets, user) for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]}
    tips = get_personalized_tips(user, targets)
    return {"type": "Weekly Plan", "targets": targets, "plan": weekly_data, "health_tips": tips}

@app.post("/monthly-plan")
async def monthly_plan(user: UserProfile):
    targets = get_targets(user)
    monthly_data = {}
    for w in range(1, 5):
        monthly_data[f"Week_{w}"] = {day: solve_day_plan(targets, user) for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]}
    tips = get_personalized_tips(user, targets)
    return {"type": "Monthly Plan", "targets": targets, "plan": monthly_data, "health_tips": tips}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)