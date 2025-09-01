import numpy as np
import pickle


# Load the pre-trained model
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


#we take arandom sample from the dataset to test the model
input_data = (6,148,72,35,0,33.6,0.627,50)

# changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#standardize the input data
# std_data = scaler.fit_transform(input_data_reshaped)
# print(f'the std data: {std_data}')

prediction = loaded_model.predict(input_data_reshaped)
print(f'The prediction is: {prediction}')

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')



