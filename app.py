import streamlit as st
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# 加载模型
try:
    model = joblib.load('rf_model.pkl')   # 确保包含 preprocessor + rf（无 SMOTE）
except FileNotFoundError:
    st.error("Model file 'rf_model.pkl' not found. Please upload the model file.")
    st.stop()

# 定义特征名称（顺序需与训练时完全一致）
feature_names = [
    "Operation method",   # 分类: 0=SPTX, 1=TPTX, 2=TPTX+AT
    "iPTH_T1",            # 连续
    "iPTH_T2",            # 连续
    "TPV",                # 连续
    "BonePain",           # 连续
    "P_T0"                # 连续
]

# 特征输入范围与类型
feature_ranges = {
    "Operation method": {
        "type": "categorical",
        "options": ["SPTX (0)", "TPTX (1)", "TPTX+AT (2)"],
        "mapping": {"SPTX (0)": 0, "TPTX (1)": 1, "TPTX+AT (2)": 2}
    },
    "iPTH_T1": {"type": "numerical", "min": 0.0, "max": 5000.0, "default": 100.0, "step": 5.0},
    "iPTH_T2": {"type": "numerical", "min": 0.0, "max": 5000.0, "default": 100.0, "step": 5.0},
    "TPV":     {"type": "numerical", "min": 0.0, "max": 10.0, "default": 1.0, "step": 0.1},
    "BonePain": {"type": "numerical", "min": 0.0, "max": 10.0, "default": 0.0, "step": 0.5},
    "P_T0":    {"type": "numerical", "min": 0.0, "max": 20.0, "default": 3.0, "step": 0.1},
}

st.set_page_config(layout="wide")
st.title("🔮 术后早期并发症预测工具 + SHAP 力图解释")
st.markdown("请输入以下 **6 项** 临床指标：")

# 收集用户输入
feature_values = {}
col1, col2 = st.columns(2)
for i, (feature, props) in enumerate(feature_ranges.items()):
    with col1 if i % 2 == 0 else col2:
        if props["type"] == "numerical":
            feature_values[feature] = st.number_input(
                label=f"{feature}",
                min_value=float(props["min"]),
                max_value=float(props["max"]),
                value=float(props["default"]),
                step=props.get("step", 1.0),
                help=f"范围: {props['min']} - {props['max']}"
            )
        else:  # categorical
            selected_label = st.selectbox(
                label=f"{feature}",
                options=props["options"],
                help="手术方式"
            )
            feature_values[feature] = props["mapping"][selected_label]

# 按正确顺序创建 DataFrame
input_df = pd.DataFrame([[feature_values[f] for f in feature_names]], columns=feature_names)

# 预测与 SHAP 力图
if st.button("🚀 开始预测", type="primary"):
    with st.spinner("计算中..."):
        try:
            # 预测概率（正类：并发症）
            proba = model.predict_proba(input_df)[0]
            risk_prob = proba[1]          # 并发症概率值
            prob_percent = risk_prob * 100

            # ---------- 风险分层（阈值 0.33 和 0.67）----------
            low_th = 0.33
            high_th = 0.67
            if risk_prob < low_th:
                risk_level = "🔵 低风险"
                risk_color = "blue"
            elif risk_prob < high_th:
                risk_level = "🟡 中风险"
                risk_color = "orange"
            else:
                risk_level = "🔴 高风险"
                risk_color = "red"

            # 显示预测结果
            st.subheader("📊 预测结果")
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.markdown(f"### 并发症风险等级：<span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
            with col_res2:
                st.metric("并发症发生概率", f"{prob_percent:.2f}%")

            # ---------- SHAP 力图解释 ----------
            st.subheader("🔍 预测解释 (SHAP Force Plot)")
            st.markdown("下图显示每个特征对预测结果的贡献：**红色**箭头推高并发症风险，**蓝色**箭头降低风险。")

            # 获取模型内部的预处理器和随机森林
            if hasattr(model, "named_steps") and "rf" in model.named_steps:
                rf_model = model.named_steps["rf"]
                preprocessor = model.named_steps.get("preprocessor", None)
            else:
                rf_model = model
                preprocessor = None

            # 预处理输入
            if preprocessor is not None:
                X_input = preprocessor.transform(input_df)
                if hasattr(X_input, "toarray"):
                    X_input = X_input.toarray()
            else:
                X_input = input_df.values

            # 计算 SHAP 值
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(X_input)

            # 提取正类的 SHAP 值（第一个样本）
            if isinstance(shap_values, list):
                shap_values_class1 = shap_values[1][0]   # 一维数组，长度=特征数
            else:
                # shap_values 形状 (n_samples, n_features, n_classes)
                shap_values_class1 = shap_values[0, :, 1]

            # 获取特征名称（清洗前缀）
            if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
                raw_names = preprocessor.get_feature_names_out()
                clean_names = [re.sub(r'^(num|cat)_+', '', n).lstrip('_') for n in raw_names]
            else:
                clean_names = feature_names

            # 获取基准值（正类）
            expected_value = explainer.expected_value
            if isinstance(expected_value, list):
                base_value = expected_value[1]
            else:
                base_value = expected_value

            # 生成 SHAP 力图（matplotlib 静态版本）
            shap.initjs()
            fig = plt.figure(figsize=(14, 4))
            shap.force_plot(
                base_value,
                shap_values_class1,
                X_input[0],
                feature_names=clean_names,
                matplotlib=True,
                show=False
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # 可选：显示全局特征重要性（用条形图）
            with st.expander("📈 查看全局特征重要性 (SHAP 平均绝对值)"):
                st.info("如需完整全局 SHAP 图，可在训练时计算并保存。")
        except Exception as e:
            st.error(f"预测过程中出错: {str(e)}")
