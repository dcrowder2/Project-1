from sklearn import preprocessing
import numpy as np
import os

os.chdir(r"C:\Users\dakota\PycharmProjects\CSCEA415Project1")

with_letters = []
with open('letter-recognition.data') as file:
    for line in file:
        temp = [i for i in line.split(',')]
        with_letters.append(temp)

without_letters = []
temp_names = []
for item in with_letters:
    temp_names.append([ord(item[0])-64])
    temp = [float(i) for i in item[1:]]
    without_letters.append(temp)
matrix = np.array(without_letters)
min_max_scaler = preprocessing.MinMaxScaler()
matrix_minmax = min_max_scaler.fit_transform(matrix)
index = 0
matrix_minmax = np.insert(matrix_minmax, [0], temp_names, axis=1)
matrix_minmax = matrix_minmax[matrix_minmax[:, 0].argsort()]
np.savetxt("normalized_data.txt", matrix_minmax)

