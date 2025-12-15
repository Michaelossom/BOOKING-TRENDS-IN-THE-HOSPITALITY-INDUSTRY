import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Hotel Booking Prediction",
    page_icon="🏨",
    layout="wide"
)

# --- LOAD MODELS ---
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('best_gb.joblib')
        scaler = joblib.load('scaler.joblib')
        model_columns = joblib.load('model_columns.joblib')
        return model, scaler, model_columns
    except FileNotFoundError:
        return None, None, None

model, scaler, model_columns = load_artifacts()

# --- MAIN PAGE HEADER ---
st.title("🏨 Hotel Haven: Cancellation Predictor")
st.markdown("""
<div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin-bottom: 20px;'>
    <p style='margin:0;'>Enter the booking details below to predict the likelihood of cancellation.</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠ Model files not found! Please run the training script (Step 1) to generate .joblib files.")
    st.stop()

# --- INPUT FORM ---
with st.form("booking_form"):
    st.subheader("📋 Booking Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        lead_time = st.number_input("Lead Time (Days)", min_value=0, max_value=500, value=50, help="Days between booking and arrival")
        avg_price = st.number_input("Avg Price per Room ($)", min_value=0.0, value=100.0)
    
    with col2:
        market_segment = st.selectbox("Market Segment", ["Online", "Offline", "Corporate", "Aviation", "Complementary"])
        special_requests = st.slider("No. of Special Requests", 0, 5, 0)

    with col3:
        res_date = st.date_input("Reservation Date", datetime.date.today())
        repeated = st.selectbox("Repeated Guest?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    st.divider()
    
    st.subheader("👥 Guest & Stay Details")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        adults = st.number_input("Adults", 1, 5, 2)
    with col2:
        children = st.number_input("Children", 0, 5, 0)
    with col3:
        week_nights = st.number_input("Week Nights", 0, 10, 2)
    with col4:
        weekend_nights = st.number_input("Weekend Nights", 0, 5, 1)

    st.divider()

    st.subheader("📝Preferences")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        meal_plan = st.selectbox("Meal Plan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"])
    with col2:
        room_type = st.selectbox("Room Type", ["Room_Type 1", "Room_Type 2", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"])
    with col3:
        car_parking = st.selectbox("Car Parking Required?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    # Submit Button
    submitted = st.form_submit_button("🚀 Predict Cancellation Risk", use_container_width=True)

# --- PREDICTION LOGIC ---
if submitted:
    # 1. Create Dataframe from inputs
    data = {
        'number of adults': adults,
        'number of children': children,
        'number of weekend nights': weekend_nights,
        'number of week nights': week_nights,
        'type of meal': meal_plan,
        'car parking space': car_parking,
        'room type': room_type,
        'lead time': lead_time,
        'market segment type': market_segment,
        'repeated': repeated,
        'P-C': 0, # Defaulting to 0 as it wasn't in main inputs
        'P-not-C': 0, # Defaulting to 0
        'average price': avg_price,
        'special requests': special_requests,
        'reservation_year': res_date.year,
        'reservation_month': res_date.month,
        'reservation_day': res_date.day,
        'reservation_dow': res_date.weekday()
    }
    
    input_df = pd.DataFrame([data])
    
    # 2. One-Hot Encoding
    input_df_encoded = pd.get_dummies(input_df)
    
    # 3. Align with model columns
    input_final = pd.DataFrame(columns=model_columns)
    for col in input_df_encoded.columns:
        if col in input_final.columns:
            input_final.loc[0, col] = input_df_encoded.iloc[0][col]
    input_final = input_final.fillna(0)
    
    # 4. Scale
    input_scaled = scaler.transform(input_final)
    
    # 5. Predict
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1] # Prob of Not Canceling
    prob_cancel = 1 - probability

    # 6. Display Results
    st.divider()
    st.subheader("📊 Prediction Results")
    
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        if prediction[0] == 0: # Canceled
            st.error("🚨 Status: *Likely to Cancel*")
            st.metric("Cancellation Probability", f"{prob_cancel:.1%}")
        else:
            st.success("✅ Status: *Likely to Stay*")
            st.metric("Stay Probability", f"{probability:.1%}")
            
    with col_res2:
        st.write("*Analysis & Recommendation:*")
        if prediction[0] == 0:
            st.warning(f"This booking has a high risk ({prob_cancel:.1%}) of cancellation. Consider sending a re-confirmation email or offering a small incentive to confirm the stay.")
            st.progress(prob_cancel)
        else:
            st.info(f"This booking is safe with a {probability:.1%} confidence level. No immediate action required.")
            st.progress(probability)
