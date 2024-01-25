'''Script to analyse and ML the CNI data from the respective CY weights'''
#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score as Acc
from sklearn.metrics import matthews_corrcoef as MCC

#%% #Import data (Weights and CNI)
Weight, CNI = [], []
with open('Data/Topological_Data.txt','r') as file:
    for idx, line in enumerate(file.readlines()[1:]):
        if idx%6 == 0: Weight.append(eval(line))
        if idx%6 == 3: CNI.append(eval(line))
Weight = np.array(Weight)
CNI = np.array(CNI)
del(file,line,idx)

#%% #Data analysis
#Plot histogram of CNI data
plt.figure('CNI Histogram')
plt.hist(CNI,bins=np.array(range(max(CNI)+2))-0.5)
plt.xlabel('CN Invariant')
plt.xlim(0,48)
plt.xticks(2*np.array(range(24))+1)
plt.ylabel('Frequency')
plt.ylim(0)
plt.grid()
plt.tight_layout()
plt.savefig('CNI_histogram.pdf')

#Plot histogram of weights
plt.figure('Weight Histogram')
plt.hist(Weight.flatten(),bins=np.array(range(max(Weight.flatten())+2))-0.5,histtype='step')
plt.xlabel('Weights')
plt.xlim(0,1000)
plt.ylabel('Frequency')
plt.ylim(0)
plt.grid()
plt.tight_layout()
#plt.savefig('Weights_histogram.pdf')

#Plot correlation matrix
df = pd.DataFrame(np.append(Weight,np.expand_dims(CNI,-1),axis=1))
corr = df.corr()
print('CNI correlations:',list(corr.loc[5,:])[:-1]) #...just focus on final row with CNI correlations with each of the 5 weights
#corr.style.background_gradient(cmap='coolwarm', axis=None).set_precision(2)

#%% #Setup data
cnis = []      #...select the cni classes to perform the classification on; empty for the full data
k = 5          #...number of k-fold cross-validations to perform (k = 5 => 80(train) : 20(test) splits approx.)
#Zip input and output data together
if len(cnis) > 0:
    ML_data = []
    for i in range(len(CNI)):
        if CNI[i] in cnis: ML_data.append([Weight[i],CNI[i]])
else:
    ML_data = [[Weight[i],CNI[i]] for i in range(len(CNI))]

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

#%% #Run NN train & test --> Classifier
#Define measure lists
Accs, MCCs = [], []    
#seed = 1                          

#Loop through each cross-validation run
for i in range(k):
    print(f'NN {i+1} training...')
    #Define & Train NN Regressor
    nn = MLPClassifier((16,32,16), activation='relu', solver='adam')#, random_state=seed)
    nn.fit(Train_inputs[i], Train_outputs[i]) 

    #Compute NN predictions on test data, and calculate learning measures
    Test_pred = nn.predict(Test_inputs[i])
    Accs.append(Acc(Test_outputs[i],Test_pred,normalize=True))
    MCCs.append(MCC(Test_outputs[i],Test_pred))

#Output averaged learning measures with standard errors
print('####################################')
print('Average Measures:')
print('Accuracy: ',sum(Accs)/k,'\pm',np.std(Accs)/np.sqrt(k))
print('MCC:      ',sum(MCCs)/k,'\pm',np.std(MCCs)/np.sqrt(k))

#%% #Run NN train & test --> Regressor
#Define measure lists
Accs, MCCs, AccsBounded = [], [], []
Rsqs = []
seed = 10                  

def round_to_nearest_odd(number):
    rounded_number = round(number)
    if rounded_number % 2 == 0:
        if rounded_number > number:
            rounded_number -= 1
        else:
            rounded_number += 1
    return rounded_number        

#Loop through each cross-validation run
for i in range(k):
    #Define & Train NN Regressor
    nn = MLPRegressor((16,32,16), activation='relu', solver='adam')#, max_iter=1000, random_state=seed)
    nn.fit(Train_inputs[i], Train_outputs[i]) 

    #Compute NN predictions on test data, and calculate learning measures
    Rsqs.append(nn.score(Test_inputs[i],Test_outputs[i]))
    Test_pred = np.array([round_to_nearest_odd(i) for i in nn.predict(Test_inputs[i])])
    #Test_pred = np.round(nn.predict(Test_inputs[i]))
    Accs.append(Acc(Test_outputs[i],Test_pred,normalize=True))
    MCCs.append(MCC(Test_outputs[i],Test_pred))
    '''
    #Compute the accuracy metric based on being within a bound that can be set
    bound = 3
    Test_pred = nn.predict(Test_inputs[i])
    AccsBounded.append(np.size(np.where(np.abs(Test_outputs[i]-Test_pred) < bound))/len(Test_pred)) #accuracy within a range
    '''

#Output averaged learning measures with standard errors
print('####################################')
print('Average Measures:')
print('R^2: ',sum(Rsqs)/k,'\pm',np.std(Rsqs)/np.sqrt(k))
print('Accuracy:    ',sum(Accs)/k,'\pm',np.std(Accs)/np.sqrt(k))
print('MCC:         ',sum(MCCs)/k,'\pm',np.std(MCCs)/np.sqrt(k))
#print('Acc-bounded: ',sum(AccsBounded)/k,'\pm',np.std(AccsBounded)/np.sqrt(k),'\n...bound:',bound)

#%% #Plot visualization of true data vs predicted data
'''
#Predict on all weight data
predicted_values = nn.predict(Weight)
fig = plt.figure()
plt.scatter(CNI, predicted_values, s=0.2, alpha=0.1)
plt.xlabel("True")
plt.ylabel("Predicted")
#plt.savefig("TruePred_all.pdf")'''

#Predict on only the test data
predicted_values = nn.predict(Test_inputs[-1])
fig = plt.figure()
plt.scatter(Test_outputs[-1], predicted_values, s=0.2, alpha=0.9)
plt.xlabel("True")
plt.ylabel("Predicted")
#plt.savefig("TruePred_testonly.pdf")

#################################################################################
#%% #Extract polynomials with the new CNI values 27 & 35
c27, c35 = [], []
with open('Data/invariants.txt','r') as file:
    for idx, line in enumerate(file.readlines()[1:]):
        if   idx%5 == 0: ww = eval(line)
        elif idx%5 == 1: pp = line.strip()
        elif idx%5 == 3: 
            CNI = eval(line)
            if   CNI == 27: c27.append([ww,pp])
            elif CNI == 35: c35.append([ww,pp])
del(file,line,idx)
print(f'CNI = 27 example: {c27[0]}')
print(f'CNI = 35 example: {c35[0]}')