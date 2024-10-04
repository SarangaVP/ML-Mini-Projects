import streamlit as st
import pandas as pd
import pickle
import numpy as np

with open('car_price_model.pkl', 'rb') as f:
    model, label_encoders = pickle.load(f)

car_data = pd.read_csv('./dataset/Carsale_data.csv')
car_data.drop(columns=['Car_ID'], inplace=True)

st.title("Car Sales Price Prediction")

st.sidebar.header('Input Features')

def user_input_features():
    brand = st.sidebar.selectbox('Brand', car_data['Brand'].unique())
    model = st.sidebar.selectbox('Model', car_data['Model'].unique())
    year = st.sidebar.slider('Year', int(car_data['Year'].min()), int(car_data['Year'].max()), int(car_data['Year'].mean()))
    fuel_type = st.sidebar.selectbox('Fuel Type', car_data['Fuel_Type'].unique())
    transmission = st.sidebar.selectbox('Transmission', car_data['Transmission'].unique())
    owner_type = st.sidebar.selectbox('Owner Type', car_data['Owner_Type'].unique())
    kilometers_driven = st.sidebar.slider('Kilometers Driven', float(car_data['Kilometers_Driven'].min()), float(car_data['Kilometers_Driven'].max()), float(car_data['Kilometers_Driven'].mean()))
    mileage = st.sidebar.slider('Mileage', float(car_data['Mileage'].min()), float(car_data['Mileage'].max()), float(car_data['Mileage'].mean()))
    engine = st.sidebar.slider('Engine', float(car_data['Engine'].min()), float(car_data['Engine'].max()), float(car_data['Engine'].mean()))
    power = st.sidebar.slider('Power', float(car_data['Power'].min()), float(car_data['Power'].max()), float(car_data['Power'].mean()))
    seats = st.sidebar.slider('Seats', int(car_data['Seats'].min()), int(car_data['Seats'].max()), int(car_data['Seats'].mean()))

    data = {'Brand': brand,
            'Model': model,
            'Year': year,
            'Kilometers_Driven': kilometers_driven,
            'Fuel_Type': fuel_type,
            'Transmission': transmission,
            'Owner_Type': owner_type,
            'Mileage': mileage,
            'Engine': engine,
            'Power': power,
            'Seats': seats}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

for column in ['Brand', 'Model', 'Fuel_Type', 'Transmission', 'Owner_Type']:
    df[column] = label_encoders[column].transform(df[column])

st.subheader('User Input features')
st.write(df)

prediction = model.predict(df)
st.subheader('Prediction')
st.write(f'Estimated Price: ${prediction[0]:.2f}')