import streamlit as st

user_input = st.text_input("label goes here")


st.write("The the privded SQL query is:", user_input)
