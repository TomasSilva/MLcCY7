'''Script to compute the CY link topological data for a variety of polynomials per weight system'''
#Import libraries
import numpy as np
import math
import re
from fractions import Fraction
from copy import deepcopy as dc

#Import data
weights, polys = [], []
with open('./Data/CYPolynomials.txt', 'r') as f: ###filepath
    for line_idx, line in enumerate(f.readlines()):
        if line_idx % 3 == 0: weights.append(eval(line))
        if line_idx % 3 == 1: polys.append(line.strip('\n'))

example_idx = 6   #...index of the example weight system in the list
example_permutation = True #...select whether to just consider the example weight system permuted, or a selection of weight systems unpermuted
num_weights = 10  #...number of weight vectors to consider
num_polys = 50    #...number of polys to generate per weight vector

#Generate permutations of the weight vector to run
if example_permutation:
    example_weights = [np.random.permutation(weights[example_idx]).tolist() for i in range(num_weights)] #...select all weight systems to be permutations of the chosen example
else:
    example_weights = [weights[i] for i in np.random.choice(range(len(weights[:1000])),num_weights,replace=False)] #...sample all weight systems randomly

#Generate initial polynomials for each weight system
init_polys = []
for ww in example_weights:
    d = sum(ww)
    degrees = str(ww).replace('[','{').replace(']','}').replace(' ','') #...reformat the weights input
    #Generate initial polynomial (basis monomials summed with coefficients all 1)
    macaulay2('r = CC[x_0..x_4, Degrees=>'+degrees+']') #...define the ring of polynomials (note this is only the coefficient ring, ground ring still CC)
    basis = macaulay2('basis('+str(d)+', r)')
    poly = str(basis).replace('| ','').replace(' |','').replace(' ','+')
    poly = re.sub(r'(\d)x',r'\1*x',poly)
    init_polys.append(poly.replace('^','**'))
    
    #Compute the singular locus dimension of the polynomial over the finite field with prime 101
    rr = macaulay2('rr = ZZ/'+str(101)+'[x_0..x_4, Degrees=>'+degrees+']') #...define the reduced ring of polynomials (note again this is only the coefficient ring, ground ring still CC)
    mac2poly = macaulay2('mac2poly = '+str(poly))
    SLdim = int(macaulay2('dim singularLocus(ideal(mac2poly))'))
    #Flag dimension errors if they occur (can add functionality from CY3PolynomialGeneration.sage script to regenerate here but currently so improbable can just rerun script if occurs)
    if SLdim > 0:
        print('error: ',ww)

################################################################################
#Generate many polynomials per weight system
prime = 101

manypolys = []
manypolys_weights = []  
for idx in range(num_weights):
    #Save the first (weight,poly) pair
    manypolys_weights.append(example_weights[idx])
    manypolys.append([init_polys[idx]])
    #Initialise new polynomial
    poly_og = dc(init_polys[idx])
    monomials = poly_og.split('+')  
    
    degrees = str(example_weights[idx]).replace('[','{').replace(']','}').replace(' ','') #...reformat the weights input
    rr = macaulay2('rr = ZZ/'+str(prime)+'[x_0..x_4, Degrees=>'+degrees+']') #...define the ring of polynomials (note this is only the coefficient ring, ground ring still CC)
    
    while len(manypolys[-1]) < num_polys:
        coef = np.random.choice(list(range(prime)),len(monomials),replace=True) #...edit new coefficient ranges here
        polynew = [str(int(coef[i]))+'*'+monomials[i] for i in range(len(monomials)) if coef[i]!=0]
        polynew = '+'.join(polynew)
        
        mac2poly = macaulay2('mac2poly = '+str(polynew.replace('**','^')))
        dimension = int(macaulay2('dim singularLocus(ideal(mac2poly))'))
        if dimension > 0: 
            continue
            
        manypolys[-1].append(polynew)
        
#print(manypolys_weights,'\n',manypolys)

################################################################################
#Run the topological invariant computation
gbls, shs, cnis = [], [], []
for w_idx in range(len(manypolys)):
    ww = manypolys_weights[w_idx]
    w_sum = sum(ww)
    gbls.append([])
    shs.append([])
    cnis.append([])
    
    for poly in manypolys[w_idx]:
        #Compute data
        stord = "Wp"+ "("+ str(ww).replace("[","").replace("]","") +")"
        A = singular.ring(0, '(x_0,x_1,x_2,x_3,x_4)', stord)
        I = singular.ideal(singular(poly))
        ra = I.jacob()
        ra = ra.groebner()
        gbl = int(singular.size(ra))
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
        
        #Save the invariants
        gbls[-1].append(gbl)
        shs[-1].append(H)
        cnis[-1].append(nu)

#Print the full data
print(gbls)
print(cnis)
print(shs)

#Print the summary checks
if example_permutation:
    print(np.unique(np.array(gbls,dtype='int32'))) #...these should be constant within sublists (same for same permutation), but vary between lists (different for different permutations)
    print(np.unique(np.array(cnis,dtype='int32'))) #...these should all be the same such that only one number shows
    print(np.unique(np.array(shs,dtype='int32').reshape((-1,2)),axis=0)) #...these should all be the same such that only one list of numbers shows
else:
    #Both of these should evaluate to True
    cnis = np.array(cnis)
    shs = np.array(shs)
    print(np.all([(cnis[i] == cnis[i,0]).all() for i in range(len(cnis))]))
    print(np.all([(len(np.unique(shs[i],axis=0)) == 1) for i in range(len(shs))]))

