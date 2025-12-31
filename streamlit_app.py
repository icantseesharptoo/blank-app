import streamlit as st

st.title("My First ML App")
st.write(
    "Experiments on Streamlit"
)

if st.button("Celebrate!"):
    st.balloons()