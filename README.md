# paper-DMS-inference

# Overview

This repository contains data and scripts for reproducing the results accompanying the manuscript

### popDMS infers mutation effects from deep mutational scanning data
Zhenchen Hong<sup>1</sup>, and John P. Barton<sup>2,#</sup>

<sup>1</sup> Department of Physics and Astronomy, University of California, Riverside  
<sup>2</sup> Department of Computational and Systems Biology, University of Pittsburgh School of Medicine  
<sup>#</sup> correspondence to [jpbarton@pitt.edu](mailto:jpbarton@pitt.edu)


# Contents

Scripts for generating and analyzing the simulation data can be found in the `WF_Simulation.ipynb` notebook. Scripts for processing and analyzing the deep mutational scanning data are contained in the `MPL_inference.ipynb` notebook. Finally, scripts for analysis and figures contained in the manuscript are located in the `figures.ipynb` notebook.  

Due to the large size and number of some files generated by the interim analysis of deep mutational scanning data, some data has been stored in a compressed format using Zenodo. To access the full set of data, navigate to the [Zenodo record](https://zenodo.org/XXXXX). Then download and extract the contents of the archives `XXXX` into the folders `data/XXXX` respectively.


# MPL

This repository includes codes for inferring selection coefficients using the marginal path likelihood (MPL) method. Codes implementing MPL in python3 and C++ are located in the `src` directory.

### Software dependencies

Here's an example statement about the need for external software to execute any part of the code: Parts of the analysis are implemented in C++11 and the [GNU Scientific Library](https://www.gnu.org/software/gsl/). 

Download Eigen package with the link: https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip to access to the C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms. Make sure that we are using the version of 3.4.0. Unzip the directory under ./epistasis_inference/. 

Figures reproduction requires one external package `logomaker` from PyPI using the pip package manager by executing the following at the commandline: `pip install logomaker`

### Epistasis inference: 
1. open the terminal and open the directory ./epistasis_inference/ 
2. enter the command line "g++ -std=c++11 -lgslcblas -lgsl -I ./eigen-3.4.0/ get_freq.cpp -o get_freq".
3. enter the command line "./get_freq"
4. enter the command line "g++ -std=c++11 -lgslcblas -lgsl -I ./eigen-3.4.0/ inversion.cpp -o inversion".
5. enter the command line "./inversion"

# License

This repository is dual licensed as [GPL-3.0](LICENSE-GPL) (source code) and [CC0 1.0](LICENSE-CC0) (figures, documentation, and our presentation of the data).