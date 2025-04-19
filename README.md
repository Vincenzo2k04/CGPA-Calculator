# Student Lifestyle & Grade Prediction

This project analyzes a dataset of students' daily habits to predict academic **Grades** using machine learning. The goal is to understand how lifestyle factors influence student performance.

## 📂 Dataset

The dataset includes 2000 entries with the following columns:

- `Study_Hours_Per_Day`
- `Extracurricular_Hours_Per_Day`
- `Sleep_Hours_Per_Day`
- `Social_Hours_Per_Day`
- `Physical_Activity_Hours_Per_Day`
- `Stress_Level` (Low, Moderate, High)
- `Gender` (Male, Female)
- `Grades` (Target Variable)

## 🧹 Data Preprocessing

1. **Label Encoding**: Categorical values (`Stress_Level`, `Gender`) encoded into numerical form.
2. **Outlier Removal**: Outliers detected and removed using the IQR (Interquartile Range) method.
3. **Normalization**: Features scaled using `StandardScaler` for better model performance.

## 🤖 Model Used

- **Random Forest Regressor**: Chosen for its robustness with tabular, mixed-type data and ability to handle non-linear relationships.

## 📊 Evaluation Metrics

- **Mean Squared Error (MSE)**: ~0.283  
- **R² Score**: ~0.446

This indicates the model explains around 44.6% of the variation in student grades, which is reasonable given real-world lifestyle variability.

## 🚀 Sample Prediction

```python
sample_input = pd.DataFrame([{
    'Study_Hours_Per_Day': 6.0,
    'Extracurricular_Hours_Per_Day': 2.0,
    'Sleep_Hours_Per_Day': 7.5,
    'Social_Hours_Per_Day': 2.0,
    'Physical_Activity_Hours_Per_Day': 3.0,
    'Stress_Level': le_stress.transform(['Moderate'])[0],
    'Gender': le_gender.transform(['Male'])[0]
}])

sample_scaled = scaler.transform(sample_input)
predicted_grade = model.predict(sample_scaled)[0]
print(f"Predicted Grade: {predicted_grade:.2f}")

🛠 Requirements
pandas

numpy

scikit-learn

pip install pandas numpy scikit-learn

📁 Files
Student.csv – Source dataset

CGPA.ipynb – Notebook with complete pipeline

README.md – This file