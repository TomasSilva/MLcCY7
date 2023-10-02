import numpy as np
from pysr import PySRRegressor
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

#Import sasakian hodge
Sweights, SHodge = [], []
with open('Data/Topological_Data.txt','r') as file:
    for idx, line in enumerate(file.readlines()[1:]):
        if idx%6 == 0: Sweights.append(eval(line))
        if idx%6 == 2: SHodge.append(eval(line))

Sweights = np.array(Sweights)        
SHodge  = np.array(SHodge)

outputs = SHodge[:,1]

X_train, X_test, y_train, y_test = train_test_split(Sweights, outputs, test_size=0.1, random_state=42)

model = PySRRegressor(
    binary_operators=["+", "-", "*", "/"], 
    niterations = 100000,
    populations = 3*os.cpu_count(),
    ncyclesperiteration = 500,
    parsimony=0.001,
    weight_optimize = 0.001,
    maxsize = 25,
    progress=True,
    loss="L2DistLoss()",
    batching=True,
    turbo = True,
)
model.fit(X_train, y_train)

print("##################################")
print("R2:", r2_score(np.round(model.predict(X_test)), y_test))
print("MAE:", mean_absolute_error(np.round(model.predict(X_test)), y_test))
print("##################################")
