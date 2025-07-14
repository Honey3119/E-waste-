
🌱 E-WasteFlow+

A Machine Learning–Driven System for Predicting E-Waste Drop-Offs at Recycling Centers

GitHub Repository: https://github.com/Honey3119/E-wasteflow

---

📖 Overview

E-WasteFlow+ is a smart, data-driven system that predicts daily e-waste drop-offs at recycling centers.
By leveraging historical data and machine learning, the system helps recycling facilities:
- Plan resources more effectively
- Reduce operational costs
- Improve sustainability outcomes

The prediction considers:
✅ Seasonal variations
✅ Day-of-week patterns
✅ Weather conditions
✅ Special collection events

---

🧰 Features

✨ Predict drop-off quantities using Random Forest regression
✨ Simulate realistic 6-month e-waste data
✨ Analyze seasonal, daily, location, and waste type trends
✨ Visualize data and model performance
✨ Actionable insights for recycling center operations

---

🗄️ Project Structure

File/Artifact           Description
-----------------------  -----------------------------------------------------------
databasesetup.py         Creates the SQLite database schema with normalized tables and pre-populates location and waste type data.
datasimulator.py         Simulates realistic daily drop-off records (January–June 2023) incorporating seasonality, weekends, weather, and events.
analysis.py              Analyzes database records and produces statistics & insights; generates ewaste_analysis.png with key visualizations.
mlmodel.py               Prepares data, trains & evaluates Random Forest model; generates ml_predictions.png showing prediction accuracy.
ewaste_analysis.png      Visualization: seasonal, daily, location-based, and waste type trends.
ml_predictions.png       Scatter plot: actual vs predicted drop-off quantities.
Project Report.pdf       Detailed final report with results, methodology, and business impact.

---

🧑‍💻 Technical Stack

- Python 3
- SQLite (lightweight database)
- Pandas, NumPy (data handling)
- Matplotlib (visualization)
- Scikit-learn (machine learning)

---

📝 Setup & Usage

1️⃣ Clone the Repository
git clone https://github.com/Honey3119/E-wasteflow.git
cd E-wasteflow

2️⃣ Create Database
python databasesetup.py

3️⃣ Simulate Data
python datasimulator.py

4️⃣ Analyze Data
python analysis.py

5️⃣ Train & Evaluate Model
python mlmodel.py

---

📊 Key Results

✅ Seasonal Trends:
- Spring peak (50.5% of total)
- Winter: 32.4%
- Summer: 17.1%

✅ Weekly Trends:
- Weekdays average: ~23 units/day
- Weekends: ~16 units/day (30% lower)

✅ Top Waste Types:
1. Batteries
2. Smartphones
3. Small Appliances

✅ Model Performance:
- R² (test): 0.661
- Test MAE: 8.73
- Test RMSE: 11.24

✅ Most Important Prediction Factors:
- Waste type
- Location
- Day of week
- Season
- Weather conditions

---

🌍 Business Impact

- 📉 Reduced empty trips and overflow
- 👥 Better staff allocation (50% more in Spring, 30% less on weekends)
- 🛣️ Improved truck routing & storage efficiency
- ♻️ Lower emissions and higher recycling rates

---

🔮 Future Work

- Integrate live weather API and holiday calendars
- Expand to multiple cities & locations
- Build mobile app for real-time updates
- Use LSTM for advanced time-series forecasting
- Add IoT sensors for automatic monitoring

---

👥 Team

- Honey Patel (hjp83)
- Bansari Patel (bdp79)

---

🎯 E-WasteFlow+ demonstrates how data science and machine learning can help solve real-world sustainability challenges and optimize operational efficiency for recycling centers.
