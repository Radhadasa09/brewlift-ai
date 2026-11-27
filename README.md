# BrewLift AI

A US-focused health and performance tracker built with [Streamlit](https://streamlit.io/) for coffee lovers and gym-goers. Users snap or upload a photo of a drink or meal, confirm or adjust serving size (`oz`) and ingredients/add-ins, track caffeine against the **400 mg/day** FDA general adult benchmark and estimated bedtime clearance, log workout burn, and save a nightly recovery check-in.

## Features
- **Photo-to-Nutrition & Caffeine Flow:** Detect coffee drinks, pre-workouts, or gym meals, edit serving sizes (`oz`) and ingredients, and recalculate calories, caffeine, protein, and key recovery micronutrients (potassium, magnesium, calcium, iron).
- **Caffeine Half-Life & Sleep Cutoff Check:** Estimate remaining caffeine at target bedtime using a 5-hour half-life model.
- **Gym Burn & Nightly Recovery Log:** Track active calories burned during workouts alongside daily intake, digestion/comfort, and sleep readiness.
- **Gumroad Subscription Gating:** Limit free daily photo scans and unlock full access with a verified Gumroad license key.

## Local Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
