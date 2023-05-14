# Overview

This repository contains data and scripts for reproducing the results accompanying the manuscript

### popDMS infers mutation effects from deep mutational scanning data
Zhenchen Hong<sup>1</sup>, and John P. Barton<sup>2,3,#</sup>

<sup>1</sup> Department of Physics and Astronomy, University of California, Riverside  
<sup>2</sup> Department of Physics and Astronomy, University of Pittsburgh  
<sup>3</sup> Department of Computational and Systems Biology, University of Pittsburgh School of Medicine  
<sup>#</sup> correspondence to [jpbarton@pitt.edu](mailto:jpbarton@pitt.edu)


# Contents

Scripts for generating and analyzing the simulation data can be found in the `WF_Simulation.ipynb` notebook. Scripts for processing and analyzing the deep mutational scanning data are contained in the `MPL_inference.ipynb` notebook. Finally, scripts for analysis and figures contained in the manuscript are located in the `figures.ipynb` notebook.  

Due to the large size and number of some files generated by the interim analysis of deep mutational scanning data, some data has been stored in a compressed format using Zenodo. To access the full set of data, navigate to the [Zenodo record](https://zenodo.org/record/7917326#.ZFu4j-xKjzc). Then download and extract the contents of the archives into the directory `./epistasis_inference/`.


# MPL

This repository includes codes for inferring selection coefficients using the marginal path likelihood (MPL) method. Codes implementing MPL in python3 and C++ are located in the `src` directory.

### Software dependencies

Parts of the analysis are implemented in C++11 and the [GNU Scientific Library](https://www.gnu.org/software/gsl/). 

Download Eigen package with this [link](https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip) to access to the C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms. Make sure that we are using the version of 3.4.0. Unzip the directory under `./epistasis_inference/`. 

Figures reproduction requires one external package `logomaker` from PyPI using the pip package manager by executing the following at the commandline: `pip install logomaker`


### popDMS input data format

popDMS can infer the selection coefficients by either full genome sequencing data or short reads sequencing data. Full genome sequencing data means the genotype counts at each generation are available, while short reads sequencing data means only single allele counts at each generation are available.

Please check the following examples to prepare the appropriate data format used in the inference pipeline. 
1) If full genome data available, the raw data format should be like this, all generations data within single file: https://github.com/bartonlab/paper-DMS-inference/blob/main/data/raw_data/TpoR_nucleotide_count.csv. 

The first column is only for indexing. 

The second column is the nucleotide variant, describing variants with respect to the nucleotide target sequence. If only one site mutates, the format should be c.site_index WT_nucleotide>Mutant_nucleotide. For example, if the site 93 has the wild type nucleotide as T and this genotype has the mutation G at site 93, the value of the second column of this genotype is c.93T>G. If more than one site mutates, the mutations should be included within the square brackets, seperated by semicolon, c.[site_index1 WT_nucleotide1>Mutant_nucleotide1; site_index2 WT_nucleotide2>Mutant_nucleotide2; ...]. For example, if the site 8 has the wild type nucleotide as C and this genotype has the mutation T at site 8, and on the same genotype, the site 9 has the wild type nucleotide as C and this genotype has the mutation G at site 9,  the value of the second column of this genotype is c.[8C>T;9C>G]. 

The third column is the amino acid variant, describing variants with respect to the amino acid target sequence. The idea is similar to the second column, but the wildtype and mutation notation would be replaced by amino acid three letters code. 

Starting from thr forth column, the counts of each genotype are recorded for each generation. 

The data file should be renamed as `Target-protein_nucleotide_count.csv`


2) If only short reads data available(single allele available), the raw data format should be like this, pre and post generation data in seperated files: https://github.com/bartonlab/paper-DMS-inference/blob/main/data/raw_data/BG505_DNA_codoncounts.csv

The first column is the site index.

The second column is the wildtype codon of this site.

The following columns record the codon counts observed according to the codon name listed in the first row.

popDMS can also do the error correction with wildtype sequencing data. 

The data files should be renamed as `Target-protein_DNA_codoncounts.csv`(error correction data), `Target-protein_mutDNA_codoncounts.csv`(pre-selection count data) and `Target-protein_mutvirus_codoncounts.csv`(post-selection count data)



### Epistasis inference: 

The following instruction is coorporated with the example of the replicate #1 of YAP1 as the target protein.

To use the epistasis inference, open the terminal and locate in the directory `cd ./epistasis_inference/` to access the pipeline codes. 

Then enter the command line `g++ -std=c++11 -lgslcblas -lgsl -I ./eigen-3.4.0/ get_freq.cpp -o get_freq` to compile the C++ script about freqeucny extraction from the raw data files. To collect intermediate frequency data files, by entering the command line with  `./get_freq [Target-protein] [location saving the genoype counts data] [indexing file for target protein]`, which in this analysis is `./get_freq YAP1 ../outputs/epistasis/YAP1_genotype_count_rep1.csv index_matrix.csv`. This step will output several allele frequency files, such as `[Target-protein]_freq_[replicate num]_[generation_num].csv` and `[Target-protein]_multiple_allele_[replicate_num]_[generation_num].csv` within the same directory you executed the `./get_freq` command line. 

After all, the internediate allele frequencies data files would be proceeded by matrix manipulation codes. First, compling the `inversion.cpp` by enetering the following command line in the terminal within the directory of `./epistasis_inference/`: `g++ -std=c++11 -lgslcblas -lgsl -I ./eigen-3.4.0/ inversion.cpp -o inversion`, then with which enter the following execution line: `./inversion [Target-protein] [location saving the genoype counts data] [indexing file for target protein]`. In this analysis, it's `./inversion YAP1 ../outputs/epistasis/YAP1_genotype_count_rep1.csv index_matrix.csv`

Finally, you will have the outputs `[Target_protein]_epistasis_rep[replicate_num].txt` within `./epistasis_inference/` directory. In this example, the output file name is `YAP1_epistasis_rep1.txt`.

# License

This repository is dual licensed as [GPL-3.0](LICENSE-GPL) (source code) and [CC0 1.0](LICENSE-CC0) (figures, documentation, and our presentation of the data).
