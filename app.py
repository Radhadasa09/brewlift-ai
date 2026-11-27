import io
import json
import os
from datetime import date, datetime, time, timedelta
from typing import List, Literal

import requests
import streamlit as st
from PIL import Image
from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Page Config & Vibrant Fitness/Coffee Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BrewLift AI — Coffee, Macros & Gym Recovery",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .hero-banner {
        background: linear-gradient(135deg, #1E1B18 0%, #3E2723 55%, #D84315 100%);
        padding: 1.25rem 1.5rem;
        border-radius: 14px;
        color: #FFFFFF;
        margin-bottom: 1rem;
    }
    .hero-banner h2 {
        margin: 0;
        font-size: 1.65rem;
        letter-spacing: -0.3px;
    }
    .hero-banner p {
        margin: 0.35rem 0 0 0;
        color: #FBE9E7;
        font-size: 0.95rem;
    }
    .stat-card {
        background-color: #161A22;
        border: 1px solid #2A3140;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Structured Schema for Gemini Multimodal Detection
# ---------------------------------------------------------------------------
class DetectedIntake(BaseModel):
    item_name: str = Field(description="Detected coffee drink, supplement, or meal name")
    category: Literal["Coffee / Pre-Workout", "Gym Meal / Snack"]
    confidence_pct: int = Field(description="Confidence score between 50 and 99")
    estimated_oz: float = Field(description="Estimated serving size in fluid oz or weight oz")
    major_ingredients: List[str] = Field(description="Key detected ingredients or add-ins")
    # Values per 1 oz so user edits to oz recalculate cleanly
    per_oz_calories: float
    per_oz_caffeine_mg: float
    per_oz_protein_g: float
    per_oz_carbs_g: float
    per_oz_fat_g: float
    per_oz_sugar_g: float
    per_oz_potassium_mg: float
    per_oz_magnesium_mg: float
    per_oz_iron_mg: float
    per_oz_calcium_mg: float
    goal_insights: List[str] = Field(
        description="2-3 practical bullet points connecting caffeine/micronutrients to gym goal and sensitivity"
    )


# ---------------------------------------------------------------------------
# Image Compression Helper (Basic Quality to Save Bandwidth & Tokens)
# ---------------------------------------------------------------------------
def compress_image_basic(uploaded_file, max_side: int = 800, quality: int = 70) -> bytes:
    """Resizes image to max_side px and compresses as lightweight JPEG."""
    img = Image.open(uploaded_file).convert("RGB")
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Gemini Vision Analyzer + Demo Fallback
# ---------------------------------------------------------------------------
def analyze_with_gemini(
    image_bytes: bytes,
    api_key: str,
    gym_goal: str,
    sensitivity: str,
) -> dict:
    if not GENAI_AVAILABLE or not api_key:
        return get_sample_detection("Coffee" if not image_bytes else "Meal")

    client = genai.Client(api_key=api_key)
    prompt = (
        f"Analyze this food or coffee/pre-workout photo for a US gym-goer. "
        f"User primary goal: {gym_goal}. User health/sensitivity focus: {sensitivity}. "
        f"Estimate realistic US serving size in ounces (oz) and provide accurate per-1-oz "
        f"calories, caffeine (mg), macros, sugar, and recovery micronutrients (potassium, magnesium, iron, calcium)."
    )

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DetectedIntake,
            temperature=0.2,
        ),
    )
    return json.loads(response.text)


def get_sample_detection(preset: str = "Coffee") -> dict:
    if preset == "Coffee":
        return {
            "item_name": "Iced Vanilla Oat Milk Latte (Double Shot)",
            "category": "Coffee / Pre-Workout",
            "confidence_pct": 94,
            "estimated_oz": 16.0,
            "major_ingredients": [
                "Double Espresso (2 oz)",
                "Oat Milk",
                "Vanilla Syrup (2 pumps)",
                "Ice",
            ],
            "per_oz_calories": 11.5,
            "per_oz_caffeine_mg": 9.4,
            "per_oz_protein_g": 0.25,
            "per_oz_carbs_g": 1.8,
            "per_oz_fat_g": 0.35,
            "per_oz_sugar_g": 1.3,
            "per_oz_potassium_mg": 16.0,
            "per_oz_magnesium_mg": 2.2,
            "per_oz_iron_mg": 0.05,
            "per_oz_calcium_mg": 22.0,
            "goal_insights": [
                "Provides ~150 mg clean pre-workout caffeine for focus and power output.",
                "Oat milk adds ~350 mg calcium across a 16 oz cup to offset caffeine calcium excretion.",
                "Contains ~21 g quick carbs/sugar—ideal 30 mins pre-lift, or swap syrup to sugar-free if cutting.",
            ],
        }
    return {
        "item_name": "Grilled Chicken Burrito Bowl with Black Beans & Avocado",
        "category": "Gym Meal / Snack",
        "confidence_pct": 91,
        "estimated_oz": 14.0,
        "major_ingredients": [
            "Grilled Chicken Breast",
            "Cilantro Lime Rice",
            "Black Beans",
            "Avocado",
            "Salsa",
        ],
        "per_oz_calories": 44.0,
        "per_oz_caffeine_mg": 0.0,
        "per_oz_protein_g": 3.4,
        "per_oz_carbs_g": 4.2,
        "per_oz_fat_g": 1.4,
        "per_oz_sugar_g": 0.3,
        "per_oz_potassium_mg": 78.0,
        "per_oz_magnesium_mg": 8.5,
        "per_oz_iron_mg": 0.32,
        "per_oz_calcium_mg": 9.0,
        "goal_insights": [
            "Delivers ~48 g lean protein to maximize post-workout muscle protein synthesis.",
            "High potassium (~1,090 mg) and magnesium (~119 mg) replenish electrolytes lost during heavy lifting.",
            "Black beans + avocado fiber supports steady blood glucose without post-meal energy crash.",
        ],
    }


# ---------------------------------------------------------------------------
# Caffeine Half-Life & Bedtime Residual Calculator
# ---------------------------------------------------------------------------
def estimate_bedtime_caffeine(intake_logs: list, bedtime_hour: int) -> float:
    """Calculates remaining caffeine (mg) at target bedtime using 5-hour half-life."""
    today = date.today()
    bed_dt = datetime.combine(today, time(hour=bedtime_hour, minute=0))
    if bedtime_hour < 6:
        bed_dt += timedelta(days=1)

    remaining_total = 0.0
    for item in intake_logs:
        mg = item.get("caffeine_mg", 0.0)
        if mg <= 0:
            continue
        logged_dt = datetime.strptime(f"{item['date']} {item['time']}", "%Y-%m-%d %H:%M")
        hours_elapsed = max(0.0, (bed_dt - logged_dt).total_seconds() / 3600.0)
        remaining_total += mg * (0.5 ** (hours_elapsed / 5.0))
    return round(remaining_total, 1)


# ---------------------------------------------------------------------------
# Gumroad License Key Verification Helper
# ---------------------------------------------------------------------------
def verify_gumroad_license(product_id: str, license_key: str) -> tuple[bool, str]:
    if license_key.strip().upper() == "DEMO-PRO-2026":
        return True, "Demo Pro license activated!"
    if not product_id or not license_key:
        return False, "Enter both Product ID (in secrets) and License Key."
    try:
        resp = requests.post(
            "https://api.gumroad.com/v2/licenses/verify",
            data={
                "product_id": product_id,
                "license_key": license_key.strip(),
                "increment_uses_count": "false",
            },
            timeout=8,
        )
        data = resp.json()
        if data.get("success"):
            purchase = data.get("purchase", {})
            if purchase.get("refunded") or purchase.get("chargebacked"):
                return False, "This license was refunded."
            if purchase.get("subscription_cancelled_at") or purchase.get("subscription_failed_at"):
                return False, "Subscription is no longer active."
            return True, "Pro Membership verified!"
        return False, data.get("message", "Invalid license key.")
    except Exception as exc:
        return False, f"Verification error: {exc}"


# ---------------------------------------------------------------------------
# Session State Setup
# ---------------------------------------------------------------------------
defaults = {
    "is_pro": False,
    "scans_today": 0,
    "detected_item": None,
    "last_compressed_kb": 0.0,
    "intake_logs": [],
    "workout_logs": [],
    "nightly_logs": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------------------------
# Sidebar: US Fitness & Coffee Profile + Gumroad Subscription Gate
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("☕ BrewLift AI")
    st.caption("US Coffee, Pre-Workout & Gym Recovery Tracker")

    st.divider()
    st.subheader("1. Gym & Coffee Profile")
    gym_goal = st.selectbox(
        "Primary Training Goal",
        [
            "Lean Muscle & Hypertrophy",
            "Fat Loss / Cutting (Low Sugar Coffee)",
            "Strength & Power Output",
            "Endurance & Cardio Conditioning",
            "Clean Energy & Deep Sleep",
        ],
    )

    sensitivity = st.selectbox(
        "Health / Sensitivity Focus",
        [
            "Sleep & Recovery Protection",
            "Acid Reflux / GERD (Low-Acid Focus)",
            "Jitter / Cortisol Sensitivity",
            "Blood Pressure & Electrolyte Balance",
            "Iron & Calcium Optimization",
        ],
    )

    c_t1, c_t2 = st.columns(2)
    with c_t1:
        calorie_target = st.number_input("Daily kcal", 1200, 5000, 2300, step=50)
    with c_t2:
        protein_target = st.number_input("Protein (g)", 60, 350, 175, step=5)

    bedtime_hour = st.slider(
        "Target Bedtime (24h)", min_value=20, max_value=24, value=22, format="%d:00"
    )
    if bedtime_hour == 24:
        bedtime_hour = 0

    st.divider()
    st.subheader("2. Gumroad Membership")
    if st.session_state.is_pro:
        st.success("✅ **BrewLift Pro Active** — Unlimited Scans")
        if st.button("Switch to Free Tier (Test)"):
            st.session_state.is_pro = False
            st.rerun()
    else:
        remaining = max(0, 3 - st.session_state.scans_today)
        st.info(f"Free Tier: **{remaining} / 3** photo scans left today")
        license_input = st.text_input(
            "Gumroad License Key",
            placeholder="Paste key (or DEMO-PRO-2026)",
        )
        if st.button("🔓 Activate Pro Membership", use_container_width=True):
            prod_id = st.secrets.get("GUMROAD_PRODUCT_ID", "") if hasattr(st, "secrets") else ""
            ok, msg = verify_gumroad_license(prod_id, license_input)
            if ok:
                st.session_state.is_pro = True
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    with st.expander("⚙️ API Key (Optional Live Vision)", expanded=False):
        default_key = (
            st.secrets.get("GEMINI_API_KEY", "")
            if hasattr(st, "secrets")
            else os.getenv("GEMINI_API_KEY", "")
        )
        user_gemini_key = st.text_input(
            "Gemini API Key", value=default_key, type="password"
        )

    st.caption(
        "Wellness Disclaimer: Caffeine (400 mg/day FDA general adult benchmark) and "
        "nutrient metrics are informational estimates, not medical advice."
    )


# ---------------------------------------------------------------------------
# Hero Header + Live Daily Caffeine & Macro Strip
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <h2>☕ BrewLift AI — Coffee Timing, Macros & Gym Recovery</h2>
        <p>Snap your coffee, pre-workout, or post-lift meal. Confirm oz & add-ins. Protect your sleep & hit your macros.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

total_kcal = round(sum(x["calories"] for x in st.session_state.intake_logs))
total_caffeine = round(sum(x["caffeine_mg"] for x in st.session_state.intake_logs))
total_protein = round(sum(x["protein_g"] for x in st.session_state.intake_logs), 1)
total_burned = round(sum(w["calories_burned"] for w in st.session_state.workout_logs))
bedtime_caff = estimate_bedtime_caffeine(st.session_state.intake_logs, bedtime_hour)

top1, top2, top3, top4 = st.columns(4)
top1.metric("Daily Caffeine", f"{total_caffeine} / 400 mg", delta=f"{400 - total_caffeine} mg left")
top2.metric(
    "Est. Bedtime Caffeine",
    f"{bedtime_caff} mg",
    delta="Sleep Safe (<50mg)" if bedtime_caff <= 50 else "High for Sleep",
    delta_color="normal" if bedtime_caff <= 50 else "inverse",
)
top3.metric("Protein Progress", f"{total_protein} / {protein_target} g")
top4.metric("Net Calories", f"{total_kcal - total_burned} kcal", delta=f"-{total_burned} workout burn")


# ---------------------------------------------------------------------------
# Main App Tabs
# ---------------------------------------------------------------------------
tab_snap, tab_gym, tab_night, tab_records = st.tabs(
    [
        "📸 1. Snap Coffee / Meal",
        "🏋️ 2. Gym Burn Log",
        "🌙 3. Nightly Recovery Audit",
        "📋 4. Daily Logs",
    ]
)

# ===========================================================================
# TAB 1: SNAP COFFEE OR MEAL -> COMPRESS -> CONFIRM OZ & INGREDIENTS
# ===========================================================================
with tab_snap:
    col_left, col_right = st.columns([1, 1.35], gap="large")

    with col_left:
        st.subheader("Step 1: Capture or Upload Photo")
        capture_type = st.radio(
            "Source", ["Upload Photo", "Camera"], horizontal=True, label_visibility="collapsed"
        )
        photo = (
            st.file_uploader("Upload JPG/PNG/WEBP", type=["jpg", "jpeg", "png", "webp"])
            if capture_type == "Upload Photo"
            else st.camera_input("Snap your cup or plate")
        )

        compressed_bytes = None
        if photo is not None:
            compressed_bytes = compress_image_basic(photo, max_side=800, quality=70)
            st.session_state.last_compressed_kb = round(len(compressed_bytes) / 1024, 1)
            st.image(compressed_bytes, caption=f"Optimized Preview ({st.session_state.last_compressed_kb} KB)", use_container_width=True)

        can_scan = st.session_state.is_pro or (st.session_state.scans_today < 3)

        if st.button(
            "⚡ Analyze Coffee or Meal",
            type="primary",
            use_container_width=True,
            disabled=not can_scan,
        ):
            st.session_state.scans_today += 1
            with st.spinner("Detecting drink/food, caffeine & recovery nutrients..."):
                if compressed_bytes and user_gemini_key:
                    st.session_state.detected_item = analyze_with_gemini(
                        compressed_bytes, user_gemini_key, gym_goal, sensitivity
                    )
                else:
                    st.session_state.detected_item = get_sample_detection("Coffee")
            st.rerun()

        c_demo1, c_demo2 = st.columns(2)
        with c_demo1:
            if st.button("☕ Sample Coffee", use_container_width=True):
                st.session_state.detected_item = get_sample_detection("Coffee")
                st.rerun()
        with c_demo2:
            if st.button("🍗 Sample Gym Bowl", use_container_width=True):
                st.session_state.detected_item = get_sample_detection("Meal")
                st.rerun()

        if not can_scan:
            st.warning("Free daily limit (3 scans) reached. Activate Pro in the sidebar for unlimited scans.")

    with col_right:
        st.subheader("Step 2: Confirm Item, Oz & Add-Ins")
        item = st.session_state.detected_item

        if item is None:
            st.info(
                "Upload a photo or click **Sample Coffee** / **Sample Gym Bowl** to preview the interactive confirmation & nutrient engine."
            )
        else:
            st.success(
                f"Detected **{item['category']}**: **{item['item_name']}** ({item['confidence_pct']}% match)"
            )

            confirmed_name = st.text_input("Confirm or Edit Name", value=item["item_name"])

            if item["category"] == "Coffee / Pre-Workout":
                preset_oz = st.segmented_control(
                    "Quick US Cup Size (fl oz)",
                    options=[8.0, 12.0, 16.0, 20.0, 24.0],
                    default=16.0 if float(item["estimated_oz"]) not in [8.0, 12.0, 16.0, 20.0, 24.0] else float(item["estimated_oz"]),
                )
                edited_oz = st.number_input(
                    "Fine-Tune Size (fl oz)",
                    min_value=1.0,
                    max_value=64.0,
                    value=float(preset_oz or item["estimated_oz"]),
                    step=1.0,
                )
            else:
                edited_oz = st.number_input(
                    f"Confirm or Edit Portion Weight in oz (AI Estimate: {item['estimated_oz']} oz)",
                    min_value=0.5,
                    max_value=80.0,
                    value=float(item["estimated_oz"]),
                    step=0.5,
                )

            extra_options = [
                "Extra Espresso Shot (+63mg caffeine)",
                "Whole Milk",
                "Oat Milk",
                "Whey Protein Scoop (+25g protein)",
                "Flavored Syrup / Sugar",
                "Olive Oil / Butter",
                "Avocado",
                "Greek Yogurt",
            ]
            confirmed_ingredients = st.multiselect(
                "Confirm or Edit Major Ingredients / Add-Ins",
                options=list(dict.fromkeys(item["major_ingredients"] + extra_options)),
                default=item["major_ingredients"],
            )

            # Dynamic Recalculation based on confirmed oz + optional quick modifiers
            caffeine_bonus = 63.0 if "Extra Espresso Shot (+63mg caffeine)" in confirmed_ingredients and "Extra Espresso Shot (+63mg caffeine)" not in item["major_ingredients"] else 0.0
            protein_bonus = 25.0 if "Whey Protein Scoop (+25g protein)" in confirmed_ingredients and "Whey Protein Scoop (+25g protein)" not in item["major_ingredients"] else 0.0
            kcal_bonus = (120.0 if protein_bonus > 0 else 0.0) + (5.0 if caffeine_bonus > 0 else 0.0)

            kcal = round(item["per_oz_calories"] * edited_oz + kcal_bonus)
            caff = round(item["per_oz_caffeine_mg"] * edited_oz + caffeine_bonus)
            prot = round(item["per_oz_protein_g"] * edited_oz + protein_bonus, 1)
            carb = round(item["per_oz_carbs_g"] * edited_oz, 1)
            fat = round(item["per_oz_fat_g"] * edited_oz, 1)
            sugar = round(item["per_oz_sugar_g"] * edited_oz, 1)
            potassium = round(item["per_oz_potassium_mg"] * edited_oz)
            magnesium = round(item["per_oz_magnesium_mg"] * edited_oz)
            iron = round(item["per_oz_iron_mg"] * edited_oz, 1)
            calcium = round(item["per_oz_calcium_mg"] * edited_oz)

            st.markdown("#### Live Nutrition, Caffeine & Recovery Micronutrients")
            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
            r1c1.metric("Calories", f"{kcal} kcal")
            r1c2.metric("Caffeine", f"{caff} mg")
            r1c3.metric("Protein", f"{prot} g")
            r1c4.metric("Carbs / Sugar", f"{carb}g / {sugar}g")

            r2c1, r2c2, r2c3, r2c4 = st.columns(4)
            r2c1.metric("Potassium", f"{potassium} mg")
            r2c2.metric("Magnesium", f"{magnesium} mg")
            r2c3.metric("Calcium", f"{calcium} mg")
            r2c4.metric("Iron", f"{iron} mg")

            st.markdown(f"#### Personalized Insights ({gym_goal})")
            for tip in item["goal_insights"]:
                st.write(f"- {tip}")

            t_col1, t_col2 = st.columns(2)
            with t_col1:
                timing_tag = st.selectbox(
                    "Timing Context",
                    [
                        "Morning Coffee / Wake-Up",
                        "Pre-Workout Fuel",
                        "Post-Workout Recovery",
                        "Lunch / Dinner",
                        "Afternoon Pick-Me-Up",
                    ],
                )
            with t_col2:
                log_time_str = st.text_input(
                    "Time Logged (HH:MM 24h)", value=datetime.now().strftime("%H:%M")
                )

            if st.button("✅ Confirm & Save to Daily Log", type="primary"):
                st.session_state.intake_logs.append(
                    {
                        "date": str(date.today()),
                        "time": log_time_str,
                        "timing": timing_tag,
                        "category": item["category"],
                        "name": confirmed_name,
                        "oz": edited_oz,
                        "ingredients": ", ".join(confirmed_ingredients),
                        "calories": kcal,
                        "caffeine_mg": caff,
                        "protein_g": prot,
                        "potassium_mg": potassium,
                        "magnesium_mg": magnesium,
                        "calcium_mg": calcium,
                        "iron_mg": iron,
                    }
                )
                st.toast("Saved to your daily coffee & gym log!", icon="☕")
                st.rerun()


# ===========================================================================
# TAB 2: ADVANCED FEATURE — GYM WORKOUT & CALORIE BURN INPUT
# ===========================================================================
with tab_gym:
    st.header("🏋️ Log Gym Session & Active Calories Burned")
    with st.form("gym_burn_form", clear_on_submit=True):
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            workout_type = st.selectbox(
                "Training Session",
                [
                    "Strength / Hypertrophy Lifting",
                    "HIIT / Metabolic Conditioning",
                    "Zone 2 Cardio / Incline Walk",
                    "Running / Cycling",
                    "CrossFit / Functional",
                ],
            )
        with g2:
            duration_min = st.number_input("Duration (min)", 10, 240, 50, step=5)
        with g3:
            active_burn = st.number_input("Active Calories Burnt", 20, 2500, 340, step=10)
        with g4:
            fuel_state = st.selectbox(
                "Fuel State", ["Caffeinated + Fed", "Caffeinated Fasted", "Uncaffeinated"]
            )

        gym_notes = st.text_input(
            "Lift / Performance Note", placeholder="e.g., Hit PR on incline bench; great focus"
        )
        if st.form_submit_button("➕ Save Workout Burn", type="primary"):
            st.session_state.workout_logs.append(
                {
                    "date": str(date.today()),
                    "workout": workout_type,
                    "duration_min": duration_min,
                    "calories_burned": active_burn,
                    "fuel_state": fuel_state,
                    "notes": gym_notes,
                }
            )
            st.success(f"Logged {workout_type} (-{active_burn} kcal).")
            st.rerun()

    if st.session_state.workout_logs:
        st.dataframe(st.session_state.workout_logs, use_container_width=True)


# ===========================================================================
# TAB 3: NIGHTLY PERFORMANCE, CAFFEINE CLEARANCE & RECOVERY RECORD
# ===========================================================================
with tab_night:
    st.header("🌙 Nightly Recovery & Caffeine-Sleep Audit")

    tot_potassium = round(sum(x["potassium_mg"] for x in st.session_state.intake_logs))
    tot_magnesium = round(sum(x["magnesium_mg"] for x in st.session_state.intake_logs))

    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Net Calories vs Target", f"{total_kcal - total_burned} / {calorie_target}")
    n2.metric("Daily Protein Hit", f"{round((total_protein / protein_target) * 100)}%")
    n3.metric("Bedtime Caffeine Load", f"{bedtime_caff} mg")
    n4.metric("Electrolytes (K / Mg)", f"{tot_potassium} / {tot_magnesium} mg")

    if bedtime_caff > 50:
        st.warning(
            f"Estimated **{bedtime_caff} mg** of active caffeine remains at {bedtime_hour:02d}:00. "
            "Consider shifting afternoon espresso or pre-workout earlier tomorrow to protect deep sleep."
        )
    else:
        st.success(
            f"Caffeine clearance looks clean (**{bedtime_caff} mg** estimated at {bedtime_hour:02d}:00)."
        )

    with st.form("nightly_audit_form"):
        s1, s2, s3 = st.columns(3)
        with s1:
            gym_pump_rating = st.slider("Workout Energy & Pump (1–10)", 1, 10, 8)
        with s2:
            stomach_comfort = st.selectbox(
                "Stomach / Acid Comfort",
                ["Clean & Calm", "Mild Coffee Acidity", "Bloated / Heavy", "Jittery"],
            )
        with s3:
            sleep_readiness = st.selectbox(
                "Sleep Readiness Tonight",
                ["Ready for Deep Sleep", "Slightly Wired", "Restless"],
            )

        recovery_note = st.text_area(
            "Nightly Health & Recovery Log",
            placeholder="Track soreness, hydration, acid reflux triggers, or tomorrow's macro/caffeine tweak...",
        )

        if st.form_submit_button("💾 Save Nightly Health Record", type="primary"):
            st.session_state.nightly_logs.append(
                {
                    "date": str(date.today()),
                    "calories_in": total_kcal,
                    "calories_burned": total_burned,
                    "net_calories": total_kcal - total_burned,
                    "protein_g": total_protein,
                    "caffeine_mg": total_caffeine,
                    "bedtime_caffeine_mg": bedtime_caff,
                    "gym_rating": gym_pump_rating,
                    "stomach_comfort": stomach_comfort,
                    "sleep_readiness": sleep_readiness,
                    "notes": recovery_note,
                }
            )
            st.success("Nightly recovery record saved!")


# ===========================================================================
# TAB 4: SAVED DAILY RECORDS
# ===========================================================================
with tab_records:
    st.header("📋 Today's Coffee, Meal & Nightly Records")
    st.subheader("Intake Log")
    if st.session_state.intake_logs:
        st.dataframe(st.session_state.intake_logs, use_container_width=True)
    else:
        st.caption("No coffee or meals logged yet.")

    st.subheader("Nightly Recovery History")
    if st.session_state.nightly_logs:
        st.dataframe(st.session_state.nightly_logs, use_container_width=True)
    else:
        st.caption("No nightly recovery records saved yet.")
