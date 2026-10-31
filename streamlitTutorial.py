
import streamlit as st
import pandas as pd
import numpy as np

st.title("Hello, Streamlit")
st.write(":streamlit: This is your first Streamlit app")
st.text("Let's get Started")

name = st.text_input("Enter your name: ")
if st.button("Hello"):
    st.success(f"Hello, {name}!!")

df = pd.DataFrame(np.random.randn(10, 2), columns=['A', 'B'])
st.line_chart(df)
st.bar_chart(df)

upload_file = st.file_uploader("Upload a CSV", type="csv")
if upload_file:
    df = pd.read_csv(upload_file)
    st.dataframe(df)

st.title("Text and Markdown Demo")
st.header("This is a header")
st.subheader("This is a subheader")
st.markdown("""
**Bold**, *italic*, `Code`, [Link](https://www.google.com)
""")
st.code("for i in range(5): print(i)", language="python")

st.text_input("What's your name?", value="35")
st.text_area("Write something...")
st.number_input("Pick a number", min_value=8, max_value=37, value=36)
st.slider("Choose a range", 8, 100, 30)
st.selectbox("Select a fruit", ["Apple", "Banana", "Tangerine"])
st.multiselect("House toppings", ["Cheese", "Tomato"])
st.radio("Pick one", ["Option A", "Option B"])
st.checkbox("I agree to the terms")

if st.checkbox("Show Details"):
    st.info("Here are more details...")

option = st.radio("Choose View", ["Show Chart", "Show Table"])
if option == "Show Chart":
    st.write("Chart would appear here")
else:
    st.write("Table would appear here")

with st.form("login_form"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    submitted = st.form_submit_button("Login")

    if submitted:
        st.success(f"Welcome, {username}")

st.image(
    "https://via.placeholder.com/300",
    caption="Sample Image",
    use_container_width=True
)
st.video("https://www.youtube.com/watch?v=VqgUkExPvLY")
