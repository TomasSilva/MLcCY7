# MLcCY7
Repository for generation of Calabi-Yau links from wp4 spaces, computation of their topological properties (Sasakian Hodge numbers, CN invariant), and their ML.

The `Data` folder lists the 7555 weights which create wp4 spaces which admit CY3 hypersurfaces in file `WP4s.txt` (the respective CY Hodge numbers (h11,h21) are listed in `WP4_Hodges.txt`). CY polynomials (with the required singularity structure to create a CY link) are given in `CYPolynomials.txt`, as well as the respective topological invariant data and Grobner bases in the corresponding files.    

The `CY3PolynomialGeneration.sage` script details how the CY3s where generated, ensuring the correct singularity structure. The `CYLinkInvariantComputation.sage` script details how the respective topological invariants were computed (code is parallelised for efficiency).    

The remaining scripts perform the respective ML investigations as detailed in the paper.       

