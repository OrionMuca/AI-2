import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
dataset = pd.read_csv(url, header=None)

# Separate features (X) and target labels (y)
X = dataset.iloc[:, :-1].values  # Features are all columns except the last one
y = dataset.iloc[:, -1].values  # Features are all columns except the last one

# Convert the DataFrame into a matrix of numbers
data = dataset.values

# Define the model
model = Sequential([
    Dense(8, input_dim=8, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=150, batch_size=16)

# Evaluate the model
loss, accuracy = model.evaluate(X, y)
print("Loss:", loss)
print("Accuracy:", accuracy)
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy * 100))
