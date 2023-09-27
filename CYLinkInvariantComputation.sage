'''Script to compute all CY link topological data (parallelised)'''
#Import libraries
import math
from fractions import Fraction
from multiprocessing import Pool

#Define function to compute all topological invariants
def TopoData(ii):
    ww, poly = ii[0], ii[1]
    w_sum = sum(ww)
    stord = "Wp"+ "("+ str(ww).replace("[","").replace("]","") +")"
    A = singular.ring(0, '(x_0,x_1,x_2,x_3,x_4)', stord)
    I = singular.ideal(singular(poly))
    ra = I.jacob()
    ra = ra.groebner()
    grobner_length = len(list(ra))
    K = ra.kbase()
    s = singular.size(K)
    
    #Compute Hodge numbers
    d_aux_list = list(singular.leadexp(poly))
    d =  sum([x*y for x,y in zip(d_aux_list,ww)])
    aux_dim = [4*d-w_sum, 3*d-w_sum]
    H=[]
    for dim in aux_dim:
        sub = []
        for i in range(1, int(s)+1):
            if sum([x*y for x,y in zip(list(singular.leadexp(K[i]))[:-1],ww)]) == dim:
                sub.append(K[i])
        H.append(len(sub))
    
    #Compute CNIs
    B = []
    for i in range(1,int(s)+1):
        temp = 0
        for j in range(0,len(ww)):
            temp += Fraction(int((1+int(K[i].leadexp()[j+1]))*ww[j]), w_sum)
        B.append(temp)
    mu_plus, mu_minus, mu_zero  = 0, 0, 0
    for i in range(0,len(B)):
        B[i] = float(B[i])
        if (B[i].is_integer() == False and math.floor(B[i]) % 2 == 0):
            mu_plus += 1
        elif (B[i].is_integer() == False and math.floor(B[i]) % 2 == 1):
            mu_minus += 1
        else:
            mu_zero += 1
    nu = (1 + len(B) - 3*(mu_plus - mu_minus))%48
    
    return(ww, poly, grobner_length, H, nu, str(list(ra))) 

################################################################################
if __name__ == '__main__':
    #Import data
    weights, polys = [], []
    with open('CYPolynomials.txt', 'r') as f: ###filepath
        for line_idx, line in enumerate(f.readlines()):
            if line_idx % 3 == 0: weights.append(eval(line))
            if line_idx % 3 == 1: polys.append(line.strip('\n'))

    # Initialise text file
    with open('./Topological_Data.txt','w') as file:
        file.write('Data: Weights, Polynomial, Hodge #s, CN invariant\n')
    # Initialise text file
    with open('./GrobnerBasis_Data.txt','w') as file:
        file.write('Data: Weights, Polynomial, GrobnerBasis, KBasis\n')

    #Run the computation
    inputs = list(zip(weights,polys))
    with Pool(48) as p:
        for idx, output in enumerate(p.imap_unordered(TopoData,inputs)):
            #Write data to respective files
            with open('./Topological_Data.txt','a') as file:
                file.write(f'{output[0]}\n{output[1]}\n{output[2]}\n{output[3]}\n{output[4]}\n\n')
            with open('./GrobnerBasis_Data.txt','a') as file:
                file.write(f'{output[0]}\n{output[1]}\n{output[5]}\n\n')

