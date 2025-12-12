# M5 Forecasting — Full Project Documentation

## 📘 Overview
This project focuses on solving the **M5 Forecasting Accuracy competition** using advanced data engineering, feature engineering, and machine learning models.  
The goal is to predict **28 days of future sales** for **30,490 items** across **3 U.S. states**, using hierarchical forecasting and the **WRMSSE metric**, which evaluates forecast quality across all hierarchy levels.

---

# 📊 WRMSSE Evaluation Metric

WRMSSE (Weighted Root Mean Squared Scaled Error) is computed as:

\[
WRMSSE = \sum_{i=1}^{42,840} w_i * RMSSE_i
\]

Each RMSSE is weighted by the importance of that series in the total sales hierarchy.

---

# 🏬 M5 Hierarchy Structure

The dataset follows a deep hierarchy:

- **3 States** → CA, TX, WI  
- **10 Stores**  
- **3 Categories**  
- **7 Departments**  
- **30,490 Items**

The forecasting must work accurately at **all 12 hierarchy levels**, including:
- Item level  
- Department level  
- Category level  
- Store level  
- State level  
- All combinations of the above  

---

# 🗂️ Dataset Description

### 1️⃣ **calendar.csv**
Contains daily information:
- Events (e.g., SuperBowl, Easter)
- SNAP (food stamp program) indicators
- Week, month, year metadata

### 2️⃣ **sell_prices.csv**
Contains historical prices for each item:
- store_id  
- item_id  
- sell_price  
- wm_yr_wk (key for joining with calendar)

### 3️⃣ **sales_train_validation.csv**
Contains **1913 days** of sales for all items.

### 4️⃣ **sales_train_evaluation.csv**
Contains **1941 days** — last 28 days used for evaluation.

---

# 🧹 Data Preprocessing

### ✔️ Step 1 — Convert Wide Format to Long Format (Melt Operation)
Wide format columns:  
`d_1, d_2, ..., d_1913`

We convert them into:

| id | item_id | d | sales |
|----|---------|---|--------|
| HOBBIES_1_001_CA_1_validation | HOBBIES_1_001 | d_1 | 0 |
| ... | ... | ... | ... |

This structure is necessary for:
- Merging with calendar data  
- Time-series feature engineering  
- WRMSSE calculation  

---

# 🔗 Merging Data

### Merge 1 — df_final with calendar  
```python
data = df_final.merge(df, on="d", copy=False)
```

### Merge 2 — Merge with sell_prices  
```python
data = data.merge(df2, on=["store_id", "item_id", "wm_yr_wk"], copy=False)
```

Final dataset is saved as:
```
final_dataframe.csv
```

---

# 🧠 Feature Engineering

Some key features include:

### 📅 Time-based Features
- Day of week  
- Week of year  
- Month  
- Year  
- Event_name_1, event_name_2  

### 📈 Lag Features
- lag_7, lag_28  
- rolling_mean_7  
- rolling_std_30  

### 🔪 Price Features
- price change %  
- rolling avg price  

---

# 🤖 Modeling

Models tested:
- **LightGBM** (best performer)
- **XGBoost**
- **CatBoost**
- **Naive and Seasonal Naive Baselines**

Target variable:
```
sales (shifted for forecasting)
```

---

# 📤 Final Submission

The model predicts:
- Day 1914 → Day 1941 (28 days future)

Submission format:
```
id, F1, F2, ..., F28
```

---

# 🧪 Key Learnings

- Large datasets require **efficient melting + chunk processing**
- WRMSSE requires **hierarchical scaling**
- Price data is extremely important
- Calendar events heavily impact predictions
- Using MySQL helps manage large CSVs efficiently

---

# 🏁 Conclusion

This project delivers:
- A complete M5 forecasting pipeline  
- Robust preprocessing  
- Feature engineering  
- Hierarchical evaluation using WRMSSE  
- Final model predictions  

---

# 📦 File Structure

```
├── bulk_load_m5_to_mysql.py
├── db_connect.py
├── EDA.ipynb
├── FeatureEngineering.ipynb
├── Modeling.ipynb
├── melted_chunks/
├── final_dataframe.csv
├── final_dataframe_test.csv
├── README.md
```

---

# 🙌 Acknowledgements

- Kaggle M5 Forecasting competition  
- Walmart Sales Dataset  
- LightGBM, XGBoost communities  

