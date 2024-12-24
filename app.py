import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Load Dataset (Replace with your actual dataset)
# For demonstration, I'll create a dummy dataset. Replace it with your own.
# Example columns: 'Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'Ips', 'PPI', 'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os', 'Price'
data = {
    'Company': ['Dell', 'HP', 'Apple', 'Lenovo', 'Asus'],
    'TypeName': ['Ultrabook', 'Notebook', 'Ultrabook', 'Notebook', 'Gaming'],
    'Ram': [8, 16, 8, 4, 32],
    'Weight': [1.2, 2.5, 1.3, 2.1, 2.8],
    'Touchscreen': [1, 0, 1, 0, 1],
    'Ips': [1, 0, 1, 0, 1],
    'PPI': [226, 141, 226, 111, 200],
    'Cpu brand': ['Intel', 'AMD', 'Intel', 'Intel', 'AMD'],
    'HDD': [0, 512, 0, 1024, 0],
    'SSD': [512, 0, 256, 0, 1024],
    'Gpu brand': ['Nvidia', 'AMD', 'Intel', 'Intel', 'Nvidia'],
    'os': ['Windows', 'Windows', 'MacOS', 'Windows', 'Windows'],
    'Price': [70000, 60000, 120000, 40000, 150000]
}
df = pd.DataFrame(data)

# Step 2: Preprocess Data
X = df.drop('Price', axis=1)
y = np.log(df['Price'])  # Using log transformation to handle large price ranges

# Encoding categorical features
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Save the model and DataFrame
with open('pipe.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('df.pkl', 'wb') as f:
    pickle.dump(df, f)

# Step 4: Load Model and Use in Streamlit App
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Price Predictor")

# UI Inputs
company = st.selectbox('Brand', df['Company'].unique())
type = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight of the Laptop')
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
screen_size = st.number_input('Screen Size')
resolution = st.selectbox(
    'Screen Resolution',
    ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440']
)
cpu = st.selectbox('CPU', df['Cpu brand'].unique())
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', df['Gpu brand'].unique())
os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    # Feature Engineering
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    # Query Construction
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
    query_df = pd.DataFrame([query], columns=['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'Ips', 'PPI', 'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os'])
    query_df = pd.get_dummies(query_df, drop_first=True).reindex(columns=X.columns, fill_value=0)

    # Prediction
    prediction = np.exp(pipe.predict(query_df)[0])
    st.title(f"The predicted price of this configuration is ₹{int(prediction)}")
