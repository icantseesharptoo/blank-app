import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.title("My First ML App")
st.write("Experiments on Streamlit")

if st.button("Celebrate!"):
    st.balloons()


@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    return pd.read_csv(url)


@st.cache_resource
def train_model(df):
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df["species"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    return model, scaler, acc


st.header("Public dataset: Iris")
df = load_data()
st.dataframe(df)

# Train model (cached)
model, scaler, acc = train_model(df)
st.subheader("Trained model")
st.write(f"Test accuracy: {acc:.3f}")

# Prediction inputs in the sidebar
st.sidebar.header("Make a prediction")
default_sl = float(df["sepal_length"].mean())
default_sw = float(df["sepal_width"].mean())
default_pl = float(df["petal_length"].mean())
default_pw = float(df["petal_width"].mean())

sepal_length = st.sidebar.number_input("Sepal length", value=default_sl, format="%.2f")
sepal_width = st.sidebar.number_input("Sepal width", value=default_sw, format="%.2f")
petal_length = st.sidebar.number_input("Petal length", value=default_pl, format="%.2f")
petal_width = st.sidebar.number_input("Petal width", value=default_pw, format="%.2f")

if st.sidebar.button("Predict"):
    X_new = [[sepal_length, sepal_width, petal_length, petal_width]]
    X_new_scaled = scaler.transform(X_new)
    pred = model.predict(X_new_scaled)[0]
    st.sidebar.success(f"Predicted species: {pred}")