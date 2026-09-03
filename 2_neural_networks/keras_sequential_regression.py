#!/usr/bin/env python
# coding: utf-8

# Sequential API is for a plain stack of layers where each layer
#   has exactly one input tensor and one output tensor

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.datasets import boston_housing

# Step 1: Prepare/load dataset
# x has 13 features (columns) with 404 training examples (rows)
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

x_train_df = pd.DataFrame(x_train)
print(x_train_df.shape)

print('Sample Data:')
print(x_train_df.head())
print('\nData Summary:')
print(x_train_df.describe())
print('\nData Null Check:')
print(x_train_df.isnull().sum())

# Step 1.1: Normalize the data (if needed)
# since the values of the features have a wide range the data should be normalize
print('Data mean by features:')
train_mean = x_train_df.mean()
print(train_mean)
print('\nData standard deviation by features:')
train_std = x_train_df.std()
print(train_std)

print('\nSample Normalized Data:')
x_train_norm = (x_train_df - train_mean) / train_std
print(x_train_norm.head())

n_cols = x_train.shape[1]

# Step 2: Build network
model = Sequential()
model.add(Input(shape=(n_cols,)))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

# Step 3: Compile the network
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Step 4: Train the model
model.fit(x_train_norm, y_train, epochs=100, verbose=2)

# Step 5: Evaluate the model's performance
x_test_df = pd.DataFrame(x_test)
print(x_test_df.shape)

x_test_norm = (x_test_df - train_mean) / train_std
print(x_test_norm.head())

model.evaluate(x_test_norm, y_test)

# Step 6: Make predictions
predictions = model.predict(x_test_norm)

# Check predictions
comparison = pd.DataFrame({'actual': y_test, 'predicted': predictions.flatten()})
print(comparison.head(10))

# Check model architecture
print(model.layers)
print(model.weights)
model.summary()
