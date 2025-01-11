import streamlit as st 
import pickle
import pandas as pd
from tensorflow.keras.models import load_model

model = load_model('brestcancerdetection.keras')
with open('scaller.pkl' , 'rb') as file:
    scaller = pickle.load(file)

st.title('Breast Cancer Detetion')

radius_mean = st.number_input("Enter Radius Mean" , min_value=0. ,max_value=50.0,format="%.6f")
texture_mean = st.number_input('Enter Texture Mean', min_value=0. ,max_value=50.0,format="%.6f")
perimeter_mean = st.number_input('Enter Perimeter Mean' ,min_value=0.0 ,max_value=200.0,format="%.6f")
area_mean = st.number_input('Enter Area Mean' , min_value=0.0, max_value=3000.0 ,format="%.6f")
smoothness_mean = st.number_input('Enter Smoothness Mean' , min_value=0.0000 , max_value=1.0000 , format="%.6f")
compactness_mean = st.number_input('Enter Compactness Mean' , min_value=0.0 , max_value=1.0 , format="%.6f")
concavity_mean = st.number_input('Enter Concavity Mean' , min_value=0.0 , max_value=1.0 , format="%.6f")
concave_points_mean =  st.number_input('Enter Concave Points Mean', min_value=0.0 , max_value=1.0 , format="%.6f")
symmetry_mean =  st.number_input('Enter Symmetry Mean', min_value=0.0 , max_value=1.0 , format="%.6f")
fractal_dimension_mean = st.number_input('Enter Fractal Dimension Mean' , min_value=0.0 , max_value=1.0 , format="%.6f")
radius_se = st.number_input('Enter Radius', min_value=0.0 , max_value=5.0 , format="%.6f")
texture_se = st.number_input('Enter Texture', min_value=0.0 , max_value=5.0 , format="%.6f")
perimeter_se = st.number_input("Enter Perimeter" , min_value=0. ,max_value=50.0,format="%.6f")
area_se = st.number_input('Enter Area SE', min_value=0. ,max_value=550.0,format="%.6f")
smoothness_se = st.number_input('Enter Smoothness SE' ,min_value=0.0 , max_value=1.0 , format="%.6f" )
compactness_se = st.number_input('Enter Compactness SE' ,min_value=0.0 , max_value=1.0 , format="%.6f" )
concavity_se = st.number_input('Enter Concavity SE' ,min_value=0.0 , max_value=1.0 , format="%.6f" )
concave_points_se = st.number_input('Enter Concave points SE' ,min_value=0.0 , max_value=1.0 , format="%.6f" )
symmetry_se = st.number_input('Enter Symmetry SE' ,min_value=0.0 , max_value=1.00 , format="%.6f" )
fractal_dimension_se = st.number_input('Enter Fractal Dimension SE' ,min_value=0.0 , max_value=1.00 , format="%.6f" )
radius_worst = st.number_input('Enter Radius Worst' ,min_value=0.0 , max_value=50.0 , format="%.6f" )
texture_worst = st.number_input('Enter Texture worst' ,min_value=0.0 , max_value=50.0 , format="%.6f" )
perimeter_worst = st.number_input('Enter Perimeter worst' ,min_value=0.0 , max_value=300.0 , format="%.6f" )
area_worst = st.number_input('Enter Area Worst' ,min_value=0.0 , max_value=5000.0 , format="%.6f" )
smoothness_worst = st.number_input('Enter Smoothness Worst' ,min_value=0.0 , max_value=1.0 , format="%.6f" )
compactness_worst = st.number_input('Enter Compactness Worst' ,min_value=0.0 , max_value=1.0 , format="%.6f" )
concavity_worst = st.number_input('Enter Concavity Worst' ,min_value=0.0 , max_value=1.0 , format="%.6f" )
concave_points_worst = st.number_input('Enter Concave Points Worst' ,min_value=0.0 , max_value=1.0 , format="%.6f" )
symmetry_worst = st.number_input('Enter Symmetry Worst' ,min_value=0.0 , max_value=1.0 , format="%.6f" )
fractal_dimension_worst = st.number_input('Enter Fractal Dimension Worst' ,min_value=0.0 , max_value=1.0 , format="%.6f" )

analysis = st.button('Detect Cancer' , type='primary' ,use_container_width=True )
if analysis:
    input_data ={
    'radius_mean': radius_mean,
    'texture_mean': texture_mean,
    'perimeter_mean': perimeter_mean,
    'area_mean': area_mean,
    'smoothness_mean': smoothness_mean,
    'compactness_mean': compactness_mean,
    'concavity_mean': concavity_mean,
    'concave points_mean': concave_points_mean,
    'symmetry_mean': symmetry_mean,
    'fractal_dimension_mean': fractal_dimension_mean,
    'radius_se': radius_se,
    'texture_se': texture_se,
    'perimeter_se': perimeter_se,
    'area_se': area_se,
    'smoothness_se': smoothness_se,
    'compactness_se': compactness_se,
    'concavity_se': concavity_se,
    'concave points_se': concave_points_se,
    'symmetry_se': symmetry_se,
    'fractal_dimension_se': fractal_dimension_se,
    'radius_worst': radius_worst,
    'texture_worst': texture_worst,
    'perimeter_worst': perimeter_worst,
    'area_worst': area_worst,
    'smoothness_worst': smoothness_worst,
    'compactness_worst': compactness_worst,
    'concavity_worst': concavity_worst,
    'concave points_worst': concave_points_worst,
    'symmetry_worst': symmetry_worst,
    'fractal_dimension_worst': fractal_dimension_worst
    }
    input_data_df = pd.DataFrame([input_data])
    scalled_data = scaller.transform(input_data_df)
    prediction = model.predict(scalled_data)
    if prediction[0][0] > 0.5:
        st.info('Cancer type is Malignant')
    else:
        st.info('Cancer type is Benign')