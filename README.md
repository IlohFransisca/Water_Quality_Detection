# 💧 Water Quality Predictor

A machine learning web application that predicts whether water is safe to drink based on its chemical and physical properties. The app combines a **Random Forest Classifier** with **rule-based WHO/EPA health and aesthetic standards** to provide a comprehensive water safety assessment.

## Live Demo
**[Click here to view the live app](https://waterqualitydetection-2bzz8kqhenqkgd96gj5emw.streamlit.app/)**

---

## About the Author
**Iloh Fransisca Onyinyechukwu**  
*Data Scientist Enthusiast*  
I am passionate about using data to solve real-world problems. This app is a blend of my machine learning skills and my interest in making water safety information accessible to everyone.

- **Email:** [ilohfransisca2014@gmail.com](mailto:ilohfransisca2014@gmail.com)
- **LinkedIn:** [ilohfransisca](https://www.linkedin.com/in/fransisca-iloh)
- **GitHub:** [ilohfransisca](https://github.com/ilohfransisca)

---

## About the Dataset
The model was trained on the **Kaggle Water Potability Dataset**, which contains 3,276 water samples with 9 features:
- `ph`: pH value of water
- `Hardness`: Capacity of water to precipitate soap (mg/L)
- `Solids`: Total dissolved solids (ppm)
- `Chloramines`: Amount of chloramines (ppm)
- `Sulfate`: Amount of sulfates (mg/L)
- `Conductivity`: Electrical conductivity (μS/cm)
- `Organic_carbon`: Amount of organic carbon (ppm)
- `Trihalomethanes`: Amount of trihalomethanes (μg/L)
- `Turbidity`: Measure of light emitting properties of water (NTU)

**Target:** `Potability` (1 = Safe to drink, 0 = Not safe)

---

## Project Methodology

### 1. Data Cleaning & Preprocessing
- **Missing Values:** Handled missing values in `ph` (15%), `Sulfate` (24%), and `Trihalomethanes` (5%) using **median imputation** to avoid the influence of outliers.
- **Anomaly Detection:** Identified and replaced a `ph = 0` value with NaN (since pH 0 is highly acidic and likely an error) before imputation.
- **Scaling:** Applied `StandardScaler` to normalize feature ranges, improving model convergence.

### 2. Exploratory Data Analysis (EDA)
- **Class Imbalance:** The target variable is imbalanced (61% Not Safe, 39% Safe).
- **Feature Analysis:** Correlation matrix revealed **no strong linear relationships** between any single feature and potability. Boxplots showed significant overlap between safe and unsafe water samples.
- **Conclusion:** A non-linear model (Random Forest) was required to capture complex feature interactions.

### 3. Modeling & Evaluation
- **Models Tested:** Logistic Regression (Baseline), Random Forest, and XGBoost.
- **Best Model:** **Random Forest** achieved the best performance (Accuracy: ~66.5%, ROC-AUC: 0.67).
- **Hyperparameter Tuning:** Utilized `GridSearchCV` to find optimal parameters (`n_estimators=200`, `max_depth=15`, `min_samples_leaf=4`).
- **Threshold Tuning (Safety-First Approach):** 
  - I deliberately chose a prediction threshold of **0.50** instead of the F1-optimized threshold of 0.35. 
  - *Reasoning:* In a water safety application, a **False Positive** (predicting unsafe water as safe) poses a severe health risk. Threshold 0.50 minimizes false positives (66 vs. 131 at threshold 0.35), prioritizing human safety over statistical accuracy.

---

## 🚀 App Features

This Streamlit application goes beyond a standard machine learning prediction by incorporating a **dual-layer assessment**:

1. **Machine Learning Prediction:** Uses the trained Random Forest model to calculate the probability of water being safe based on complex feature patterns.
2. **Rule-Based Standards Evaluation (WHO/EPA):** 
   - **Health-Based Limits:** Checks for Chloramines, Sulfate, Trihalomethanes, Turbidity, and pH against established health limits.
   - **Aesthetic Guidelines:** Flags issues with Total Dissolved Solids, taste, odor, and scaling potential.
3. **Interactive Sidebar:** Includes author information, app details, a feedback form, and direct contact links.
4. **Detailed Prediction Insights:** Users can expand a section to see the raw probability scores and the model threshold used.

---

## Repository Structure

```text
water_quality_detection/
├── app.py                     # Main Streamlit application code
├── water_quality_model.pkl    # Trained Random Forest model
├── scaler.pkl                 # Fitted StandardScaler
├── threshold.txt              # Saved optimal threshold (0.50)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation

## How to Run Locally
1. Clone the repository:
   git clone https://github.com/ilohfransisca/water_quality_detection.git
   cd water_quality_detection
2. Install the required packages:
   pip install -r requirements.txt
3. Run the Streamlit app:
   streamlit run app.py
4. Open your browser:
   The app will automatically open at http://localhost:8501.

## Limitations & Disclaimer
   Model Accuracy: The dataset is notoriously challenging, with published research achieving 65-70% accuracy. Our model achieves ~66% accuracy. It is designed as a screening tool, not a certified laboratory test.

   Averages: EDA revealed that average values for safe and unsafe water are almost identical, making it difficult for any model to separate them perfectly.

   Disclaimer: This app is a machine learning prediction combined with rule-based guidelines, not a certified water test. Always verify with proper laboratory testing before drinking.

## Acknowledgements
Dataset Source: Kaggle Water Potability

Built with: Streamlit, Scikit-Learn, Pandas, NumPy
