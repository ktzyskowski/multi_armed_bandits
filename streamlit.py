import streamlit as st

st.title("k-Armed Bandit Testbed")

with st.sidebar:
    k = st.number_input("Number of arms: $k$", 1, 10)

    st.checkbox("ε-Greedy")

    st.radio("Select one:", [1, 2])
