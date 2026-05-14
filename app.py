import streamlit as st
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 加载模型
try:
    model = joblib.load('rf.pkl')
except FileNotFoundError:
    st.error("Model file 'rf.pkl' not found. Please upload the model file.")
    st.stop()

# 特征范围定义
feature_names = [
    "Age", "BMI", "Diabetes", "time", "Abdominal"
]
feature_ranges = {
    "Age": {"type": "numerical", "min": 18, "max": 100, "default": 18},
    "BMI": {"type": "numerical", "min": 10, "max": 100, "default": 10},
    "Diabetes": {"type": "categorical", "options": ["0", "1"]},
    "time": {"type": "numerical", "min": 100, "max": 600, "default": 100},
    "Abdominal": {"type": "categorical", "options": ["0", "1"]},
}

# Streamlit 界面
st.title("Prediction Model with SHAP Visualization")
st.header("Enter the following feature values:")

feature_values = {}
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        feature_values[feature] = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        feature_values[feature] = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )

# 处理分类特征
label_encoders = {}
for feature, properties in feature_ranges.items():
    if properties["type"] == "categorical":
        label_encoders[feature] = LabelEncoder()
        label_encoders[feature].fit(properties["options"])
        feature_values[feature] = label_encoders[feature].transform([feature_values[feature]])[0]

# 转换为模型输入格式
features = pd.DataFrame([feature_values], columns=feature_names)

# 预测与 SHAP 可视化
if st.button("Predict"):
    try:
        # 模型预测
        predicted_class = model.predict(features)[0]
        predicted_proba = model.predict_proba(features)[0]

        # 提取预测的类别概率
        probability = predicted_proba[predicted_class] * 100

        # 显示预测结果
        st.subheader("Prediction Result:")
        st.write(f"Predicted possibility of Early complications is **{probability:.2f}%**")
    # 添加必需的 except 块
    except Exception as e:
        st.error(f"预测过程中出错: {str(e)}")
