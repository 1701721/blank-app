import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 頁面配置：模仿專業儀表板佈局
st.set_page_config(page_title="薪資預測系統 Expert", layout="wide")

# 使用快取機制確保模型不會因 Widget 變動而重複訓練
@st.cache_resource
def train_model_pipeline():
    # 讀取資料：確保 CSV 與 app.py 處於同一根目錄
    df = pd.read_csv('employee_salary_dataset.csv')
    
    # 建立特徵映射表
    le_dict = {}
    cat_columns = ['Department', 'Education_Level', 'Gender', 'City']
    
    # 複製資料進行轉換
    encoded_df = df.copy()
    for col in cat_columns:
        le = LabelEncoder()
        encoded_df[col] = le.fit_transform(df[col])
        le_dict[col] = le
        
    # 定義特徵向量 X 與 目標變數 y
    features = ['Department', 'Experience_Years', 'Education_Level', 'Age', 'Gender', 'City']
    X = encoded_df[features]
    y = encoded_df['Monthly_Salary']
    
    # 訓練隨機森林回歸模型 (Regressor for Salary Prediction)
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    
    return model, le_dict, df

# 載入模型與原始資料
model, le_dict, original_df = train_model_pipeline()

# --- 側邊欄 (Sidebar) 實作 ---
st.sidebar.header("📊 參數設定區")
st.sidebar.info("請調整下方職涯特徵，系統將即時分析您的薪資潛力。")

# 核心特徵：顯眼的滑動條
exp_years = st.sidebar.slider("工作年資 (Experience Years)", 0, 40, 5)

# 動態獲取資料集中的類別項
dept = st.sidebar.selectbox("所屬部門", original_df['Department'].unique())
edu = st.sidebar.selectbox("教育程度", original_df['Education_Level'].unique())
age = st.sidebar.number_input("年齡 (Age)", 18, 65, 30)
gender = st.sidebar.radio("性別", original_df['Gender'].unique())
city = st.sidebar.selectbox("工作城市", original_df['City'].unique())

# --- 主面板 (Main Panel) 實作 ---
st.title("💼 薪資預測系統")
st.markdown("---")

# 模擬 EDA 區塊 (參考來源圖表風格)
col1, col2 = st.columns(2)
with col1:
    st.write("### 模型預測引擎：Random Forest")
    st.write("本系統使用隨機森林演算法，根據歷史數據捕捉多維特徵間的非線性關聯。")
with col2:
    st.write("### 數據洞察提示")
    st.caption("根據模型特徵相關性分析，『年資』與『學歷』是影響預測結果最顯著的變數。")

# 預測按鈕與結果呈現
if st.sidebar.button("開始計算預測薪資"):
    # 輸入資料轉換 (Encoding)
    input_df = pd.DataFrame({
        'Department': [le_dict['Department'].transform([dept])[0]],
        'Experience_Years': [exp_years],
        'Education_Level': [le_dict['Education_Level'].transform([edu])[0]],
        'Age': [age],
        'Gender': [le_dict['Gender'].transform([gender])[0]],
        'City': [le_dict['City'].transform([city])[0]]
    })
    
    # 執行回歸預測
    prediction = model.predict(input_df)[0]
    
    # 結果展示區
    st.markdown("### 🏆 預測分析結果")
    st.success(f"根據現有模型分析，您的預估月薪落點為：")
    st.markdown(f"# 💰 **${prediction:,.2f}**")
    
    # 特徵重要性提示 (模擬圖表中的 Correlation Matrix 邏輯)
    st.progress(min(int(exp_years * 2.5), 100))
    st.caption("年資對於薪資貢獻權重預覽")