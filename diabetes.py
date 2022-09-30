import numpy as np
import pandas as pd
from perceptron import Perceptron

data = pd.read_csv("C:/Users/Jin Ni/Desktop/diabetes_scale.txt", sep = " ", header=None)
data = data.drop(data.columns[9], axis = 1)
data = data.to_numpy()
for i in range(len(data)):
    for j in range(len(data[i])):
        if j == 0:
            data[i][j] = float(data[i][j])
        else:
            try:
                data[i][j] = float(data[i][j][2:])
            except:
                data[i][j] = float(0)

data = data.astype('float64')
# np.random.shuffle(data)
training = data[:614]
trainingX = training[:, 1:]
trainingY = training[:, 0]
valid = data[614:]
validX = valid[:, 1:]
validY = valid[:, 0]

a = Perceptron()
b = Perceptron()
c = Perceptron()
preda = np.zeros(10)
predb = np.zeros(10)
predc = np.zeros(10)
a.training(trainingX, trainingY, 0.01, 100)
b.training(trainingX, trainingY, 0.01, 200)
c.training(trainingX, trainingY, 0.01, 300)
i = 0
preda[i] = np.sum(validY == a.predict(validX, 0)) / len(validY)
predb[i] = np.sum(validY == b.predict(validX, 0)) / len(validY)
predc[i] = np.sum(validY == c.predict(validX, 0)) / len(validY)

# training = data[:614]
# trainingX = training[:, 1:]
# trainingY = training[:, 0]
# valid = data[614:]
# validX = valid[:, 1:]
# validY = valid[:, 0]

print("Accuracy with 100 epoch and 0.05 learning rate:", end=" ")
print(preda[0])

print("Accuracy with 200 epoch and 0.05 learning rate:", end=" ")
print(predb[0])

print("Accuracy with 300 epoch and 0.05 learning rate:", end=" ")
print(predc[0])

a.training(trainingX, trainingY, 0.01, 400)
b.training(trainingX, trainingY, 0.01, 500)
c.training(trainingX, trainingY, 0.01, 600)
i = 0
preda[i] = np.sum(validY == a.predict(validX, 0)) / len(validY)
predb[i] = np.sum(validY == b.predict(validX, 0)) / len(validY)
predc[i] = np.sum(validY == c.predict(validX, 0)) / len(validY)

# training = data[:614]
# trainingX = training[:, 1:]
# trainingY = training[:, 0]
# valid = data[614:]
# validX = valid[:, 1:]
# validY = valid[:, 0]

print("Accuracy with 400 epoch and 0.05 learning rate:", end=" ")
print(preda[0])

print("Accuracy with 500 epoch and 0.05 learning rate:", end=" ")
print(predb[0])

print("Accuracy with 600 epoch and 0.05 learning rate:", end=" ")
print(predc[0])



