# MLcCY7
Repository for generation of Calabi-Yau links from wp4 spaces, computation of their topological properties (Sasakian Hodge numbers, CN invariant), and their ML.

The `Data` folder lists the 7555 weights which create wp4 spaces which admit CY3 hypersurfaces in file `WP4s.txt` (the respective CY Hodge numbers (h11,h21) are listed in `WP4_Hodges.txt`). CY polynomials (with the required singularity structure to create a CY link) are given in `CYPolynomials.txt`, as well as the respective topological invariant data in the file `Topological_Data.txt` listing each weight system, CY polynomial, Sasakian Hodge numbers, CN invariant, then Groebner basis length respectively. A selection of Groebner bases are given explicitly in the zipped file also.    

The `CY3PolynomialGeneration.sage` script details how the CY3s where generated, ensuring the correct singularity structure. The `CYLinkInvariantComputation.sage` script details how the respective topological invariants were computed (code is parallelised for efficiency). The `wREquivalenceChecks.sage` script contains general functionality for statistically verifying the weak R-Equivalence conjecture across the database.        

The remaining scripts perform the respective ML investigations as detailed in the paper.       

