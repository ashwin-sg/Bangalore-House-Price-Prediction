import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import pickle
import numpy as np
import json

from PIL import Image

pickle_in = open('price_model.pkl', 'rb')
model = pickle.load(pickle_in)

f = open("columns.json","r")
columns = json.loads(f.read())
locations = [i for i in columns['data_columns'][4:]]
X = pd.DataFrame(columns)


def Welcome():
    return 'WELCOME ALL!'


def predict_price(location, sqft, bath, balcony, bhk):
    """Let's Authenticate the Banks Note
    This is using docstrings for specifications.
    ---
    parameters:
      - name: location
        in: query
        type: text
        required: true
      - name: sqft
        in: query
        type: number
        required: true
      - name: bath
        in: query
        type: number
        required: true
      - name: balcony
        in: query
        type: number
        required: true
      - name: bhk
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values

    """
    x = np.zeros(245)
    x[0] = sqft
    x[1] = bath
    x[2] = balcony
    x[3] = bhk
    if 'location_' + location in X.columns:
        loc_index = np.where(X.columns == "location_" + location)[0][0]
        x[loc_index] = 1
    return model.predict([x])[0]


def main():
    st.title("Bangalore House Price Prediction")
    page_bg_img = '''
         <style>
         body {
         background-image: url("https://cdn.shopify.com/s/files/1/0285/1316/products/w0398_1s_Dashes-in-Geometric-Diamonds-Wallpaper-for-Walls-Gertrude-Stein_For-Interior-Walls-12.jpg?v=1586017533");
         background-size: cover;
         </style>


         '''

    st.markdown(page_bg_img, unsafe_allow_html=True)



    html_temp = """
    <h4 style="color:black;text-align:left;"> Using Streamlit Web Application for ML </h4>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    st.subheader('Please enter the required details:')
    location = st.selectbox("Choose Location", [i for i in locations[4:]])
    sqft = st.text_input("Area in Sq-ft", "")
    bath = st.text_input("Number of Bathroom", "")
    balcony = st.text_input("Number of Balconies", "")
    bhk = st.text_input("Number of Bedrooms", "")

    result = ""

    if st.button("House Price in Lakhs"):
        result = round(predict_price(location, sqft, bath, balcony, bhk),2)
        st.success('The House Price is {} Lakhs Indian Rupees'.format(result))
    if st.button("About"):
        st.text("Created by Ashwin SG")
        st.text("A Data Science Enthusiast")


if __name__ == '__main__':
    main()