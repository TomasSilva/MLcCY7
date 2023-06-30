'''Script to generate CY3 polynomials from the wp4 weights'''
#Import libraries
import re
import numpy as np
from copy import deepcopy as dc

#Import data
with open('./Data/WP4s.txt','r') as file:
    Weights = eval(file.read())
del(file)
print('Data imported...')  

#Compute a CY polynomial with correct singularity structure for each weight system
polynomials = []
nonzeroSLdimcount, attempt_limit, failures_count = 0, 100, 0
for wp4 in Weights[:]:
    d = sum(wp4)
    degrees = str(wp4).replace('[','{').replace(']','}').replace(' ','') #...reformat the weights input
    #Generate initial polynomial (basis monomials summed with coefficients all 1)
    macaulay2('r = CC[x_0..x_4, Degrees=>'+degrees+']') #...define the ring of polynomials (note this is only the coefficient ring, ground ring still CC)
    basis = macaulay2('basis('+str(d)+', r)')
    poly = str(basis).replace('| ','').replace(' |','').replace(' ','+')
    poly = re.sub(r'(\d)x',r'\1*x',poly)
    #Compute the singular locus dimension of the polynomial over the finite field with prime 101
    rr = macaulay2('rr = ZZ/'+str(101)+'[x_0..x_4, Degrees=>'+degrees+']') #...define the reduced ring of polynomials (note again this is only the coefficient ring, ground ring still CC)
    mac2poly = macaulay2('mac2poly = '+str(poly))
    SLdim = int(macaulay2('dim singularLocus(ideal(mac2poly))'))
    #Deal with cases where singularity structure not correct for link construction (i.e. SLdim != 0)
    if SLdim > 0:
        #Check any SLdim > 0 dimension is not due to a bad prime reduction (negligible probability of reducing badly on multiple primes, but if does still fine to regenerate a different polynomial)
        bad_prime_reduction = False
        for prime in [251, 1993, 1997]:
            rr2 = macaulay2('rr2 = ZZ/'+str(prime)+'[x_0..x_4, Degrees=>'+degrees+']') #...define another reduced ring of polynomials over a different prime
            mac2poly2 = macaulay2('mac2poly2 = '+str(poly))
            SLdim2 = int(macaulay2('dim singularLocus(ideal(mac2poly2))'))
            #Check any SLdim > 0 dimension is not due to a bad prime reduction
            if SLdim2 == 0: 
                bad_prime_reduction = True
                break
        #Regenerate a CY polynomial with the correct singularity structure
        if not bad_prime_reduction:
            nonzeroSLdimcount += 1
            rr = macaulay2('rr = ZZ/'+str(101)+'[x_0..x_4, Degrees=>'+degrees+']') #...redefine original ring over prime 101
            monomials = dc(poly.split('+'))
            attempt_counter = 0
            while SLdim > 0 and attempt_counter < attempt_limit: 
                coef = np.random.choice(list(range(1,5)),len(monomials))
                poly = [str(int(coef[i]))+'*'+monomials[i] for i in range(len(monomials))]
                poly = '+'.join(poly)
                mac2poly = macaulay2('mac2poly = '+str(poly))
                SLdim = int(macaulay2('dim singularLocus(ideal(mac2poly))'))
            #Where a SLdim == 0 polynomial couldn't be generated output an error in this case (practically only required up to 2 regenerations, so this never occurred)
            if attempt_counter == attempt_limit: 
                poly = 'error'
                failures_count += 1
    
    #Save the final polynomial (reformated to sagemath style)
    polynomials.append(poly.replace('^','**'))
    
print(f'{nonzeroSLdimcount} initial polynomials had SLdm > 0 ({failures_count} couldnt be regenerated with SLdim == 0)')

#Sort the data in increasing polynomial length
lengths = [len(poly.split('+')) for poly in polynomials]
alldata = sorted(zip(lengths,Weights,polynomials))
_, Weights_sorted, polynomials_sorted = zip(*alldata)

#Save the data
with open('CYPolynomials.txt','w') as file:
    for w_idx, wp4 in enumerate(Weights_sorted):
        file.write(f'{wp4}\n{polynomials_sorted[w_idx]}\n\n')  

