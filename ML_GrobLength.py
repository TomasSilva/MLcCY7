'''Script to ML regress the length of Grobner basis from the CY weights'''
#Import libraries
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dc
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE

#%% #Import data
weights, grobner_lengths = [], []
with open('Data/Topological_Data.txt','r') as file:
    for idx, line in enumerate(file.readlines()[1:]):
        if idx%6 == 0: weights.append(eval(line))
        if idx%6 == 4: grobner_lengths.append(eval(line))
weights = np.array(weights)
grobner_lengths = np.array(grobner_lengths)
del(file,idx,line)

#%% #Grobner length correlations
plt.figure('Histogram')
plt.hist(grobner_lengths,bins=np.array(range(max(grobner_lengths)+2))-0.5)
plt.xlabel(r'Gr$\ddot{o}$bner Basis Length') 
plt.ylabel('Frequency')
plt.ylim(0)
#plt.xlim(0,7500)
plt.grid()
plt.tight_layout()
#plt.savefig('GrobnerLengths_histogram.pdf')

#%% #Set-up data for ML
k = 5 #...number of k-fold cross-validations to perform (k = 5 => 80(train) : 20(test) splits approx.)
ML_data = list(zip(weights,grobner_lengths))

#Shuffle data ordering
np.random.shuffle(ML_data)
s = int(np.floor(len(ML_data)/k)) #...number of datapoints in each validation split
if k == 1: s = int(np.floor(0.2*len(ML_data)))
    
#Define data lists, each with k sublists with the relevant data for that cross-validation run
Train_inputs, Train_outputs, Test_inputs, Test_outputs = [], [], [], []
if k > 1:
    for i in range(k):
        Train_inputs.append([datapoint[0] for datapoint in ML_data[:i*s]]+[datapoint[0] for datapoint in ML_data[(i+1)*s:]])
        Train_outputs.append([datapoint[1] for datapoint in ML_data[:i*s]]+[datapoint[1] for datapoint in ML_data[(i+1)*s:]])
        Test_inputs.append([datapoint[0] for datapoint in ML_data[i*s:(i+1)*s]])
        Test_outputs.append([datapoint[1] for datapoint in ML_data[i*s:(i+1)*s]])
else:
     Train_inputs  = [[datapoint[0] for datapoint in ML_data[s:]]]
     Train_outputs = [[datapoint[1] for datapoint in ML_data[s:]]]
     Test_inputs   = [[datapoint[0] for datapoint in ML_data[:s]]]
     Test_outputs  = [[datapoint[1] for datapoint in ML_data[:s]]]
     
del(ML_data) #...zipped list no longer needed

#%% #Run NN train & test --> Regressor
#Define measure lists
MSEs, MAEs, MAPEs, Rsqs, Accs, NNs = [], [], [], [], [], []
bound = 0.05*(np.max(grobner_lengths)-np.min(grobner_lengths))
#seed = 1                        

#Loop through each cross-validation run
for i in range(k):
    print(f'NN {i+1} training...')
    #Define & Train NN Regressor directly on the data
    #Edit NN params bellow...
    nn_reg = MLPRegressor((16,32,16),activation='relu',solver='adam')#,random_state=seed)
    nn_reg.fit(Train_inputs[i], Train_outputs[i]) 
    NNs.append(dc(nn_reg))
    
    #Compute NN predictions on test data, and calculate learning measures
    Test_pred = nn_reg.predict(Test_inputs[i])
    Rsqs.append(nn_reg.score(Test_inputs[i],Test_outputs[i]))
    MSEs.append(MSE(Test_outputs[i],Test_pred,squared=True))   
    MAEs.append(MAE(Test_outputs[i],Test_pred))          
    MAPEs.append(MAPE(Test_outputs[i],Test_pred)) 
    Accs.append(np.mean(np.where(np.absolute(np.array(Test_outputs[i])-Test_pred) < bound,1,0)))

#Output averaged learning measures with standard errors
print('####################################')
print('Average Measures:')
print('R^2: ',sum(Rsqs)/k,'\pm',np.std(Rsqs)/np.sqrt(k))
print('MSE: ',sum(MSEs)/k,'\pm',np.std(MSEs)/np.sqrt(k))
print('MAE: ',sum(MAEs)/k,'\pm',np.std(MAEs)/np.sqrt(k))
print('MAPE:',sum(MAPEs)/k,'\pm',np.std(MAPEs)/np.sqrt(k))
print('Accuracy:',sum(Accs)/k,'\pm',np.std(Accs)/np.sqrt(k))

#%% #Predict on the remaining six weight systems
remaining_weights = np.array([[1, 1, 8, 19, 28], [1, 1, 9, 21, 32], [1, 1, 11, 26, 39], [1, 1, 12, 28, 42], [1, 6, 34, 81, 122], [1, 6, 40, 93, 140]])
GBL_predictions = []
for net in NNs:
    GBL_predictions.append([])
    for ww in remaining_weights:
        GBL_predictions[-1].append(int(np.round(nn_reg.predict([ww])[0])))
        #print(f'Weight system: {ww}\nGroebner basis length: {int(np.round(nn_reg.predict([ww])[0]))}')
GBL_predictions = np.array(GBL_predictions)

print(f'Weight systems:\n{remaining_weights}\n')
print(f'GBLs: {np.round(np.mean(GBL_predictions,axis=0))}')

##############################################################################
#%% #Train on short GB and test on long GB
#Set-up data
sorted_data = sorted(list(zip(weights,grobner_lengths)),key=lambda x: x[1])

s = int(np.floor(0.95*len(sorted_data)))
train = sorted_data[:s]
test = sorted_data[s:]

np.random.shuffle(train)
np.random.shuffle(test)

Train_inputs, Train_outputs = zip(*train)
Test_inputs,  Test_outputs  = zip(*test)

Train_inputs = np.array(Train_inputs)
Train_outputs = np.array(Train_outputs)
Test_inputs = np.array(Test_inputs)
Test_outputs = np.array(Test_outputs)

#Train NN
nn_reg = MLPRegressor((16,32,16),activation='relu',solver='adam')#,random_state=seed)
nn_reg.fit(Train_inputs, Train_outputs) 

#Test NN
Test_pred = nn_reg.predict(Test_inputs)
print(f'R^2: {nn_reg.score(Test_inputs,Test_outputs)}')
print(f'MSE: {MSE(Test_outputs,Test_pred,squared=True)}')   
print(f'MAE: {MAE(Test_outputs,Test_pred)}')          
print(f'MAPE: {MAPE(Test_outputs,Test_pred)}') 
bound = 0.05*(np.max(grobner_lengths)-np.min(grobner_lengths))
print(f'Accuracy: {np.mean(np.where(np.absolute(np.array(Test_outputs)-Test_pred) < bound,1,0))}')

##############################################################################
#%% #Print the correlation between Grobner basis length and sasakian h21 (run ML_Hodge.py cell to import Sweights & SHodge data used in this analysis)
GScombined = np.vstack((grobner_lengths,SHodge[:,1]))
print(f'PMCC: {np.corrcoef(GScombined)}')

#%% #Cross-plot Grobner length against Sh21
plt.figure()
plt.scatter(GScombined[0,:],GScombined[1,:],alpha=0.1)
#plt.axline((0, 0), slope=1, c='k')
plt.xlabel('Grobner Basis Length')
plt.ylabel('Sasakian '+r'$h^{2,1}$')
plt.grid()
plt.tight_layout()
#plt.savefig('GrobSh21_scatter.pdf')

#%% #Correlate polynomial length with GB length
weights, poly_lengths, grobner_lengths = [], [], []
with open('Data/Topological_Data.txt','r') as file:
    for idx, line in enumerate(file.readlines()[1:]):
        if idx%6 == 0: weights.append(eval(line))
        if idx%6 == 1: poly_lengths.append(len(line.strip().split('+')))
        if idx%6 == 4: grobner_lengths.append(eval(line)) ####formatted like this due to data errors
del(file,idx,line)

print(f'PMCC: {np.corrcoef(list(zip(*list(zip(poly_lengths,grobner_lengths)))))}')
