'''Script to ML regress the length of Grobner basis from the CY weights'''
#Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE

#%% #Import data
weights, grobner_lengths = [], []
with open('Data/GrobnerBasis_Data.txt','r') as file:
    for idx, line in enumerate(file.readlines()):
        if idx%4 == 0: weights.append(eval(line))
        if idx%4 == 2: grobner_lengths.append(len(line.replace('[1,','[').split(','))) ####formatted like this due to data errors
del(file,idx,line)

#%% #Grobner length correlations
plt.figure('Histogram')
plt.hist(grobner_lengths,bins=np.array(range(max(grobner_lengths)+2))-0.5)
plt.xlabel(r'Grobner Basis Length') 
plt.ylabel('Frequency')
plt.ylim(0)
plt.grid()
plt.tight_layout()
#plt.savefig('GrobnerLengths_histogram.pdf')

#%% #Set-up data for ML
k = 5          #...number of k-fold cross-validations to perform (k = 5 => 80(train) : 20(test) splits approx.)
ML_data = list(zip(weights,grobner_lengths))

#Shuffle data ordering
np.random.shuffle(ML_data)
s = int(np.floor(len(ML_data)/k)) #...number of datapoints in each validation split
if k == 1: s = int(np.floor(0.8*len(ML_data)))
    
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
MSEs, MAEs, MAPEs, Rsqs = [], [], [], []
seed = 1                        

#Loop through each cross-validation run
for i in range(k):
    print(f'NN {i+1} training...')
    #Define & Train NN Regressor directly on the data
    #Edit NN params bellow!!!
    nn_reg = MLPRegressor((256,256,256),activation='relu',solver='adam')#,random_state=seed)
    nn_reg.fit(Train_inputs[i], Train_outputs[i]) 

    #Compute NN predictions on test data, and calculate learning measures
    Test_pred = nn_reg.predict(Test_inputs[i])
    Rsqs.append(nn_reg.score(Test_inputs[i],Test_outputs[i]))
    MSEs.append(MSE(Test_outputs[i],Test_pred,squared=True))   
    MAEs.append(MAE(Test_outputs[i],Test_pred))          
    MAPEs.append(MAPE(Test_outputs[i],Test_pred)) 

#Output averaged learning measures with standard errors
print('####################################')
print('Average Measures:')
print('R^2: ',sum(Rsqs)/k,'\pm',np.std(Rsqs)/np.sqrt(k))
print('MSE: ',sum(MSEs)/k,'\pm',np.std(MSEs)/np.sqrt(k))
print('MAE: ',sum(MAEs)/k,'\pm',np.std(MAEs)/np.sqrt(k))
print('MAPE:',sum(MAPEs)/k,'\pm',np.std(MAPEs)/np.sqrt(k))

##############################################################################
#%% #Match up datasets (run ML_Hodge.py cell to import Sweights & SHodge data used in this analysis)
GScombined = []
for w1_idx in range(len(weights)):
    for w2_idx in range(len(Sweights)):
        if np.array_equal(weights[w1_idx],Sweights[w2_idx]):
            GScombined.append([grobner_lengths[w1_idx],SHodge[w2_idx][1]])
            break
    if len(GScombined) == len(Sweights): break ###remove when have full data
del(w1_idx,w2_idx)
print(f'PMCC: {np.corrcoef(GScombined.transpose())}')

#%% #Cross-plot Grobner length against Sh21
GScombined = np.array(GScombined)
plt.figure()
plt.scatter(GScombined[:,0],GScombined[:,1],alpha=0.1)
#plt.axline((0, 0), slope=1, c='k')
plt.xlabel('Grobner Basis Length')
plt.ylabel('Sasakian '+r'$h^{2,1}$')
plt.grid()
plt.tight_layout()
#plt.savefig('GrobSh21_scatter.pdf')

#%% #Correlate polynomial length with GB length
weights, poly_lengths, grobner_lengths = [], [], []
with open('Data/groebner_basis.txt','r') as file:
    for idx, line in enumerate(file.readlines()):
        if idx%4 == 0: weights.append(eval(line))
        if idx%4 == 1: poly_lengths.append(len(line.strip().split('+')))
        if idx%4 == 2: grobner_lengths.append(len(line.replace('[1,','[').split(','))) ####formatted like this due to data errors
del(file,idx,line)

print(f'PMCC: {np.corrcoef(list(zip(*list(zip(poly_lengths,grobner_lengths)))))}')
