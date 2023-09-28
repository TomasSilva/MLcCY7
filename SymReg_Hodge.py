import numpy as np
from ast import literal_eval
from pysr import PySRRegressor
import os

#Import sasakian hodge
Sweights, SHodge = [], []
with open('Data/Topological_Data.txt','r') as file:
    for idx, line in enumerate(file.readlines()[1:]):
        if idx%6 == 0: Sweights.append(eval(line))
        if idx%6 == 2: SHodge.append(eval(line))

Sweights = np.array(Sweights)        
SHodge  = np.array(SHodge)

outputs = SHodge[:,1]


model = PySRRegressor(
    binary_operators=["+", "-", "*", "/"], 
    unary_operators=["square", "cube", "sqrt"], 
    niterations = 30,
    populations = 3*os.cpu_count(),
    ncyclesperiteration = 5000,
    parsimony=0.0001,
    weight_optimize = 0.001,
    progress=True,
    loss="L2DistLoss()",
    batching=True,
    turbo = True,
)
model.fit(Sweights, outputs)
