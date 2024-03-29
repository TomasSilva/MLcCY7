'''Script to correlate CY hodge with Sasakian hodge and ML'''
#Import libraries
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dc
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE

#%% #Import data
#Import weights and CY hodge
with open('Data/WP4s.txt','r') as file:
    weights = eval(file.read())
with open('Data/WP4_Hodges.txt','r') as file:
    CYhodge = eval(file.read())
CY = [[weights[i],CYhodge[i]] for i in range(7555)]

#Import sasakian hodge
Sweights, SHodge = [], []
with open('Data/Topological_Data.txt','r') as file:
    for idx, line in enumerate(file.readlines()[1:]):
        if idx%6 == 0: Sweights.append(eval(line))
        if idx%6 == 2: SHodge.append(eval(line))
del(file,line,idx)

#%% #Match up datasets
combined = []
for w1_idx in range(7555):
    for w2_idx in range(len(Sweights)):
        if np.array_equal(CY[w1_idx][0],Sweights[w2_idx]):
            #combined.append(CY[w1_idx]+[SHodge[w2_idx]])
            combined.append([CY[w1_idx][1],SHodge[w2_idx]])
            break
    if len(combined) == len(Sweights): break ###remove when have full data
del(w1_idx,w2_idx)

#%% #Histograms of entries
#Convert to arrays so easier to slice
weights = np.array(weights)
CYhodge = np.array(CYhodge)
SHodge  = np.array(SHodge)

#Output Hodge number histograms
print(f'h^{3,0} data: {np.unique(SHodge[:,0],return_counts=True)}')
plt.figure('Histogram')
plt.hist(SHodge[:,1],bins=np.array(range(max(SHodge[:,1])+2))-0.5)
plt.xlabel(r'Sasakian $h^{2,1}$')
plt.ylabel('Frequency')
plt.ylim(0)
plt.grid()
plt.tight_layout()
#plt.savefig('Sh21_histogram.pdf')

#%% #Cross-plotting of hodge numbers
CYh_idx = 1  #...{0,1} => {h11,h21}
CYh_labels = [r'$h^{1,1}$',r'$h^{2,1}$']
Sh_idx  = 1  #...{0,1} => {h30,h21}
Sh_labels  = [r'$h^{3,0}$',r'$h^{2,1}$']
combined = np.array(combined)
print(f'CYh21 = Sh21: {len(np.where(combined[:,0,1]==combined[:,1,1])[0])}')
print(f'PMCC: {np.corrcoef(combined.reshape((len(combined),4)).transpose())}')

plt.figure()
plt.scatter(combined[:,0,1],combined[:,1,1],alpha=0.2)
#plt.axline((0, 0), slope=1, c='k') #add y=x line
plt.xlabel('CY '+CYh_labels[CYh_idx])
plt.ylabel('Sasakian '+Sh_labels[Sh_idx])
plt.grid()
plt.tight_layout()
#plt.savefig('h21s_scatter.pdf')

##############################################################################
#%% #Set-up data for ML
k = 5 #...number of k-fold cross-validations to perform (k = 5 => 80(train) : 20(test) splits approx.)

ML_data = list(zip(Sweights,SHodge[:,1]))
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
bound = 0.05*(np.max(SHodge)-np.min(SHodge))
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
    MAEs.append(MAE(Test_outputs[i],Test_pred))
    MSEs.append(MSE(Test_outputs[i],Test_pred,squared=True))             
    MAPEs.append(MAPE(Test_outputs[i],Test_pred)) 
    Accs.append(np.mean(np.where(np.absolute(np.array(Test_outputs[i])-Test_pred) < bound,1,0)))

#Output averaged learning measures with standard errors
print('####################################')
print('Average Measures:')
print('R^2: ',sum(Rsqs)/k,'\pm',np.std(Rsqs)/np.sqrt(k))
print('MAE: ',sum(MAEs)/k,'\pm',np.std(MAEs)/np.sqrt(k))
print('MSE: ',sum(MSEs)/k,'\pm',np.std(MSEs)/np.sqrt(k))
print('MAPE:',sum(MAPEs)/k,'\pm',np.std(MAPEs)/np.sqrt(k))
print('Accuracy:',sum(Accs)/k,'\pm',np.std(Accs)/np.sqrt(k))

#%% #Plot (pred v true)
true = SHodge[:,1]
pred = nn_reg.predict(Sweights)

plt.figure(figsize = (8, 8))
plt.scatter(Test_outputs,Test_pred,alpha=1,s=5) #s sets the dot size
plt.axline((0, 0), slope=1, c='k', linewidth = 0.8, linestyle = '--') #add y=x line
plt.xlabel(r'Sasakian $h^{2,1}$')
plt.ylabel(r'NN Predicted $h^{2,1}$')
plt.xlim(0,440)
plt.ylim(0,440)
plt.grid()
#plt.tight_layout()
plt.savefig('h21s_nntruepred.pdf')

#%% #Test trained Sh21 NN on CYh21 data
CY_pred = nn_reg.predict(weights)
print(f'Rsq: {nn_reg.score(weights,CYhodge[:,1])}')
print(f'MAE: {MAE(CYhodge[:,1],CY_pred)}')
print(f'Acc: {np.mean(np.where(np.absolute(CYhodge[:,1]-CY_pred) < 0.05*(np.max(CYhodge[:,1])-np.min(CYhodge[:,1])),1,0))}')

#%% #Predict on the remaining six weight systems
remaining_weights = np.array([[1, 1, 8, 19, 28], [1, 1, 9, 21, 32], [1, 1, 11, 26, 39], [1, 1, 12, 28, 42], [1, 6, 34, 81, 122], [1, 6, 40, 93, 140]])
Sh21_predictions = []
for net in NNs:
    Sh21_predictions.append([])
    for ww in remaining_weights:
        Sh21_predictions[-1].append(int(np.round(nn_reg.predict([ww])[0])))
        #print(f'Weight system: {ww}\nSasakian h21: {int(np.round(nn_reg.predict([ww])[0]))}')
Sh21_predictions = np.array(Sh21_predictions)
print(f'Weight systems:\n{remaining_weights}\n')
print(f'Sasakian h21: {np.round(np.mean(Sh21_predictions,axis=0))}')

#%% #Print the equivalent Calabi-Yau Hodge numbers
for idx in range(6):
    for ii, row in enumerate(weights):
        if np.array_equal(row,remaining_weights[idx]):
            findex = ii
            break
    print(CYhodge[ii][1])
