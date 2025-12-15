
# 🏨 Hotel Haven: Booking Cancellation Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://booking-t.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Complete-green)

## 📖 Project Overview
*Hotel Haven* is a luxury hotel chain facing a significant challenge: high and unpredictable cancellation rates. These cancellations lead to revenue loss, inefficient staffing, and suboptimal inventory management.

This project uses *Machine Learning* to predict whether a customer will cancel their booking based on reservation details (e.g., lead time, market segment, price). By identifying high-risk bookings, the hotel can implement proactive retention strategies or dynamic overbooking to maximize occupancy.

*Key Goal:* Build an end-to-end ML pipeline and a user-friendly Streamlit application to classify bookings as Canceled or Not_Canceled.

---

## 🔗 Live Demo
Check out the deployed application here: **[Hotel Haven Prediction App](https://booking-t.streamlit.app/)**

---

## 📊 Dataset
The dataset contains *36,285* observations and *17* features.
- *Source:* booking.csv
- *Target Variable:* booking status (Binary: Canceled / Not_Canceled)
- *Key Features:*
  - lead time: Days between booking and arrival (Strongest predictor).
  - average price: Average price per room.
  - special requests: Number of special requests made by the guest.
  - market segment type: How the booking was made (Online, Offline, Corporate, etc.).

---

## 🛠 Tech Stack
- *Language:* Python
- *Data Manipulation:* Pandas, NumPy
- *Visualization:* Matplotlib, Seaborn
- *Machine Learning:* Scikit-Learn (Random Forest, Gradient Boosting, KNN)
- *Deployment:* Streamlit
- *Versioning:* Git

---

## 📂 Project Structure
```text
├── app.py                   # Streamlit Application (Frontend)
├── booking.csv              # Raw Dataset
├── model_training.ipynb     # Jupyter Notebook for EDA & Model Development
├── requirements.txt         # List of dependencies
├── README.md                # Project Documentation
├── gb_model.joblib          # Saved Gradient Boosting Model
├── scaler.joblib            # Saved StandardScaler
└── model_columns.joblib     # Saved Feature Columns

