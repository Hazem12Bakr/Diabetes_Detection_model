import numpy as np
import streamlit as st
import pickle


# Load the pre-trained model
# loaded_model = pickle.load(open('D:/study/projects/AI/machine learning/Diabetes Prediction project/model.pkl', 'rb'))
loaded_model = pickle.load(open('model.pkl', 'rb'))

# Create a function for prediction
def diabetes_prediction(input_data):
    # changing the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # # standardize the input data
    # std_data = scaler.fit_transform(input_data_reshaped)
    # print(f'the std data: {std_data}')

    prediction = loaded_model.predict(input_data_reshaped)
    print(f'The prediction is: {prediction}')

    if (prediction[0] == 0):
        return 'Not Diabetic'
    else:
        return 'Diabetic'



def main():

    # Giving a title
    st.title('Diabetes Prediction Web App')

    # Getting the input data from the user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age of the Preson')

    # Code for Prediction
    diagonsis = ''

    # Creating a button for Prediction
    if st.button('Predict'):
        diagonsis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    st.success(diagonsis)


if __name__ == '__main__':
    main()
