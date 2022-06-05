import sys, os
#from copy import deepcopy
from importlib import reload

import numpy as np

import scipy as sp
import scipy.stats as st
from scipy import interpolate 

import pandas as pd
import seaborn as sns

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plot
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

import tqdm
import time
import csv
import math
import csv

# GLOBAL VARIABLES

NUC    = ['-', 'A', 'C', 'G', 'T']                           # Nucleotide letter
REF    = NUC[0]
CODONS = ['AAA', 'AAC', 'AAG', 'AAT', 'ACA', 'ACC', 'ACG', 'ACT',   # Tri-nucleotide units table
          'AGA', 'AGC', 'AGG', 'AGT', 'ATA', 'ATC', 'ATG', 'ATT',
          'CAA', 'CAC', 'CAG', 'CAT', 'CCA', 'CCC', 'CCG', 'CCT',
          'CGA', 'CGC', 'CGG', 'CGT', 'CTA', 'CTC', 'CTG', 'CTT',
          'GAA', 'GAC', 'GAG', 'GAT', 'GCA', 'GCC', 'GCG', 'GCT',
          'GGA', 'GGC', 'GGG', 'GGT', 'GTA', 'GTC', 'GTG', 'GTT',
          'TAA', 'TAC', 'TAG', 'TAT', 'TCA', 'TCC', 'TCG', 'TCT',
          'TGA', 'TGC', 'TGG', 'TGT', 'TTA', 'TTC', 'TTG', 'TTT']   
#AA  = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I','L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '*']
AA  = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '*']
NUC = ['A', 'C', 'G', 'T']
MU  = { 'GC': 1.0e-7,                                         
        'AT': 7.0e-7,
        'CG': 5.0e-7,
        'AC': 9.0e-7,
        'GT': 2.0e-6,
        'TA': 3.0e-6,
        'TG': 3.0e-6,
        'CA': 5.0e-6,
        'AG': 6.0e-6,
        'TC': 1.0e-5,
        'CT': 1.2e-5,
        'GA': 1.6e-5  }

'''
aa2codon = {                                                         # DNA codon table
    'A' : ['GCT', 'GCC', 'GCA', 'GCG'],
    'R' : ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'N' : ['AAT', 'AAC'],
    'D' : ['GAT', 'GAC'],
    'C' : ['TGT', 'TGC'],
    'E' : ['GAA', 'GAG'],
    'Q' : ['CAA', 'CAG'],
    'G' : ['GGT', 'GGC', 'GGA', 'GGG'],
    'H' : ['CAT', 'CAC'],
    'I' : ['ATT', 'ATC', 'ATA'],
    'L' : ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
    'K' : ['AAA', 'AAG'],
    'M' : ['ATG'],
    'F' : ['TTT', 'TTC'],
    'P' : ['CCT', 'CCC', 'CCA', 'CCG'],
    'S' : ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
    'T' : ['ACT', 'ACC', 'ACA', 'ACG'],
    'W' : ['TGG'],
    'Y' : ['TAT', 'TAC'],
    'V' : ['GTT', 'GTC', 'GTA', 'GTG'],
    '*' : ['TAA', 'TGA', 'TAG'],
    '-' : ['---'],
    }                                              
'''

aa2codon = {                                                         # DNA codon table
    'A' : ['GCT', 'GCC', 'GCA', 'GCG'],
    'C' : ['TGT', 'TGC'],
    'D' : ['GAT', 'GAC'],
    'E' : ['GAA', 'GAG'],
    'F' : ['TTT', 'TTC'],
    'G' : ['GGT', 'GGC', 'GGA', 'GGG'],
    'H' : ['CAT', 'CAC'],
    'I' : ['ATT', 'ATC', 'ATA'],
    'K' : ['AAA', 'AAG'],
    'L' : ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
    'M' : ['ATG'],
    'N' : ['AAT', 'AAC'],
    'P' : ['CCT', 'CCC', 'CCA', 'CCG'],
    'Q' : ['CAA', 'CAG'],
    'R' : ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'S' : ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
    'T' : ['ACT', 'ACC', 'ACA', 'ACG'],
    'V' : ['GTT', 'GTC', 'GTA', 'GTG'],
    'W' : ['TGG'],
    'Y' : ['TAT', 'TAC'],
    '*' : ['TAA', 'TGA', 'TAG'],
    '-' : ['---'],
    }                                              


def codon2aa(c, noq=False):              # Returns the amino acid character corresponding to the input codon.
        
    if c[0]=='-' and c[1]=='-' and c[2]=='-': return '-'        # If all nucleotides are missing, return gap
    
    elif c[0]=='-' or c[1]=='-' or c[2]=='-':                   # Else if some nucleotides are missing, return '?'
        if noq: return '-'
        else:   return '?'
   
    # If the first or second nucleotide is ambiguous, AA cannot be determined, return 'X'
    elif c[0] in ['W', 'S', 'M', 'K', 'R', 'Y'] or c[1] in ['W', 'S', 'M', 'K', 'R', 'Y']: return 'X'     
                                                    
    elif c[0]=='T':                                             # Else go to tree
        if c[1]=='T':
            if    c[2] in ['T', 'C', 'Y']: return 'F'
            elif  c[2] in ['A', 'G', 'R']: return 'L'
            else:                          return 'X'
        elif c[1]=='C':                    return 'S'
        elif c[1]=='A':
            if    c[2] in ['T', 'C', 'Y']: return 'Y'
            elif  c[2] in ['A', 'G', 'R']: return '*'
            else:                          return 'X'
        elif c[1]=='G':
            if    c[2] in ['T', 'C', 'Y']: return 'C'
            elif  c[2]=='A':               return '*'
            elif  c[2]=='G':               return 'W'
            else:                          return 'X'
        else:                              return 'X'
        
    elif c[0]=='C':
        if   c[1]=='T':                    return 'L'
        elif c[1]=='C':                    return 'P'
        elif c[1]=='A':
            if    c[2] in ['T', 'C', 'Y']: return 'H'
            elif  c[2] in ['A', 'G', 'R']: return 'Q'
            else:                          return 'X'
        elif c[1]=='G':                    return 'R'
        else:                              return 'X'
        
    elif c[0]=='A':
        if c[1]=='T':
            if    c[2] in ['T', 'C', 'Y']: return 'I'
            elif  c[2] in ['A', 'M', 'W']: return 'I'
            elif  c[2]=='G':               return 'M'
            else:                          return 'X'
        elif c[1]=='C':                    return 'T'
        elif c[1]=='A':
            if    c[2] in ['T', 'C', 'Y']: return 'N'
            elif  c[2] in ['A', 'G', 'R']: return 'K'
            else:                          return 'X'
        elif c[1]=='G':
            if    c[2] in ['T', 'C', 'Y']: return 'S'
            elif  c[2] in ['A', 'G', 'R']: return 'R'
            else:                          return 'X'
        else:                              return 'X'
        
    elif c[0]=='G':
        if   c[1]=='T':                    return 'V'
        elif c[1]=='C':                    return 'A'
        elif c[1]=='A':
            if    c[2] in ['T', 'C', 'Y']: return 'D'
            elif  c[2] in ['A', 'G', 'R']: return 'E'
            else:                          return 'X'
        elif c[1]=='G':                    return 'G'
        else:                              return 'X'

    else:                                  return 'X'



def err_correct(DMS_DIR, DNACODON):

    df_0 = pd.read_csv('%s/%s' % (DMS_DIR, DNACODON), comment = '#', memory_map = True)     # Read raw data file

    r    = len(CODONS)              # How many codon 
    L    = len(df_0)                # How long the sequence is
    err  = np.zeros((L, r))         # Error matrix record
    q    = len(AA)                  # How many amino acid

    # Get wildtype sequence
    wt = []
    for i in range(L):
        wt.append(str(df_0.iloc[i].wildtype))

    # Get the total number of reads for each site
    norm_0 = np.zeros(L)
    for i in range(L):
        norm_0[i] = df_0[CODONS].sum(axis=1).tolist()[i]

    #norm_0 = np.zeros(L)
    #for c in CODONS:
    #    for i in range(L):
    #        norm_0[i] += df_0.iloc[i][c]
         

    # Compute error probability
    for i in range(L):
        err[i] = df_0.iloc[i][CODONS].tolist()/norm_0[i]       
        wild_type_index = CODONS.index(wt[i])       
        err[i][wild_type_index] = norm_0[i]/df_0.iloc[i][CODONS].tolist()[wild_type_index]

    #for c in CODONS:
    #    for i in range(L):
    #        if c == wt[i]:
    #            err[i][CODONS.index(c)] = norm_0[i]/df_0.iloc[i][c]
    #        else:
    #            err[i][CODONS.index(c)] = df_0.iloc[i][c]/norm_0[i]	

    return err



def joint_replicates_list(**func_para):
    
    MPL_DIR       = func_para['MPL_DIR']
    DMS_DIR       = func_para['DMS_DIR']
    DATA_DIR      = func_para['DATA_DIR']
    MPL_RAW_DIR   = func_para['MPL_RAW_DIR']
    MPL_RAW_FILES = func_para['MPL_RAW_FILES']
    PRE_FILES     = func_para['PRE_FILES']
    POST_FILES    = func_para['POST_FILES']
    homolog       = func_para['homolog']
    regular_list  = func_para['regular_list']
    runname_index = func_para['runname_index']
    DNACODON      = func_para['DNACODON']

    print('------ Calculating joint selection coefficients for %s ------' %homolog)

    df_0 = pd.read_csv('%s/%s/%s' % (DMS_DIR, DATA_DIR, PRE_FILES[0]), comment = '#', memory_map = True)
    q    = len(AA)
    L    = len(df_0)
    size = q * L
    dx   = np.zeros(size)
    dmut = np.zeros(size)
    cov  = np.zeros((size, size))

    print("\nCalculating error probability from %s...\n" %DNACODON)
    err = err_correct(DMS_DIR, DATA_DIR, DNACODON)

    for run_name in runname_index:     # If we have more experiment replicates, the result will be more accurate.
        print("Cumulating replicate_%s data of %s..." %(run_name, homolog))
        index = runname_index.index(run_name)
        df_0 = pd.read_csv('%s/%s/%s' % (DMS_DIR, DATA_DIR, PRE_FILES[index]),  comment = '#', memory_map = True)
        df_1 = pd.read_csv('%s/%s/%s' % (DMS_DIR, DATA_DIR, POST_FILES[index]), comment = '#', memory_map = True)

        x_0  = np.zeros(size)
        x_1  = np.zeros(size)

        # Get wildtype sequence
        wt = []
        for i in range(L):
            wt.append(str(df_0.iloc[i].wildtype))
        print("Correcting reads and computing allele frequency difference and mutational contribution...")
        # Compute the total number of reads for each site after error correction      
        norm_0 = np.zeros(L)
        norm_1 = np.zeros(L)
#new code
        for i in range(L):
            temp_1_wt = df_1.iloc[i][wt[i]]
            temp_0_wt = df_0.iloc[i][wt[i]]
            wt_index = CODONS.index(wt[i])
            
            x = df_0.iloc[i][CODONS].tolist() - temp_0_wt * err[i]
            y = df_1.iloc[i][CODONS].tolist() - temp_1_wt * err[i]
            
            x = [np.max([x_i,0]) for x_i in x]
            y = [np.max([y_i,0]) for y_i in y]

            x[wt_index] = temp_0_wt*err[i][wt_index]
            y[wt_index] = temp_1_wt*err[i][wt_index]
            norm_0[i] = sum(x)
            norm_1[i] = sum(y)


            x_0_c = [x_i/norm_0[i] for x_i in x]
            x_1_c = [x_i/norm_1[i] for x_i in y]
            
            
            for c in CODONS:    
                aa    = codon2aa(c)
                aaidx = AA.index(aa)
                
                x_0[(q * i) + aaidx] += x_0_c[CODONS.index(c)]
                x_1[(q * i) + aaidx] += x_1_c[CODONS.index(c)]
                dx[(q * i) + aaidx]  += x_1_c[CODONS.index(c)] - x_0_c[CODONS.index(c)]


                for c_i in range(3):
                    for n in NUC:
                        if n!=c[c_i]:
                            m_aa    = codon2aa([c[k] if k !=c_i else n for k in range(3)])
                            m_aaidx = AA.index(m_aa)
                            dmut[(q * i) + aaidx]   -= x_0_c[CODONS.index(c)] * MU[c[c_i]+n]
                            dmut[(q * i) + m_aaidx] += x_0_c[CODONS.index(c)] * MU[c[c_i]+n]

#new code


#        for c in CODONS:
#            for i in range(L):
#                if c==wt[i]:
#                    norm_0[i] += df_0.iloc[i][c]*err[i][CODONS.index(c)]
#                    norm_1[i] += df_1.iloc[i][c]*err[i][CODONS.index(c)]
#                else:
#                    norm_0[i] += np.max([df_0.iloc[i][c] - df_0.iloc[i][wt[i]]*err[i][CODONS.index(c)], 0])
#                    norm_1[i] += np.max([df_1.iloc[i][c] - df_1.iloc[i][wt[i]]*err[i][CODONS.index(c)], 0])



        # Compute dx and mutational contribution 
#        for c in CODONS:
#            aa    = codon2aa(c)
#            aaidx = AA.index(aa)
#            for i in range(L):

                # change in frequency
#                x_0_c = np.max([df_0.iloc[i][c] - df_0.iloc[i][wt[i]]*err[i][CODONS.index(c)], 0])
#                x_1_c = np.max([df_1.iloc[i][c] - df_1.iloc[i][wt[i]]*err[i][CODONS.index(c)], 0])
#                if c==wt[i]:
#                    x_0_c = df_0.iloc[i][c]*err[i][CODONS.index(c)]
#                    x_1_c = df_1.iloc[i][c]*err[i][CODONS.index(c)]
#                x_0_c /= norm_0[i]
#                x_1_c /= norm_1[i]
#                x_0[(q * i) + aaidx] += x_0_c
#                x_1[(q * i) + aaidx] += x_1_c
#                dx[(q * i) + aaidx]  += x_1_c - x_0_c

                # mutational flux
#                for c_i in range(3):
#                    for n in NUC:
#                        if n!=c[c_i]:
#                            m_aa    = codon2aa([c[k] if k !=c_i else n for k in range(3)])
#                            m_aaidx = AA.index(m_aa)
#                            dmut[(q * i) + aaidx]   -= x_0_c * MU[c[c_i]+n]
#                            dmut[(q * i) + m_aaidx] += x_0_c * MU[c[c_i]+n]

        # Get wildtype sequence for amino acid
        wt = []
        for i in range(L):
            wt.append(AA.index(codon2aa(df_0.iloc[i].wildtype)))

        # Compute average frequencies for all states, and non-WT frequencies for each site
        x_avg = np.zeros(size)
        x_mut = np.zeros(L)
        for i in range(L):
            for a in range(q):
                x_avg[(q * i) + a] = (x_0[(q * i) + a] + x_1[(q * i) + a])/2

                if a!=wt[i]:
                    x_mut[i] += x_avg[(q * i) + a]

        # Compute covariance
        for i in range(L):
            for a in range(q):

                # diagonal terms
                cov[(q * i) + a, (q * i) + a] += x_avg[(q * i) + a] * (1 - x_avg[(q * i) + a])
                #cov[(q * i) + a, (q * i) + a] += (1/regular)/3  # diagonal regularization

                # off-diagonal, same site
                for b in range(a+1, q):
                    cov[(q * i) + a, (q * i) + b] += -x_avg[(q * i) + a] * x_avg[(q * i) + b]
                    cov[(q * i) + b, (q * i) + a] += -x_avg[(q * i) + a] * x_avg[(q * i) + b]
 
    for regular in regular_list:
        print('Calculating joint selection coefficients with regularization term = %d' %regular)
        cov_temp = cov.copy()
        for i in range(L):
            cov_block = np.zeros((q, q))
            for a in range(q):
                cov_temp[(q * i) + a, (q * i) + a] += (1/regular)  # diagonal regularization
                for b in range(q):
                    cov_block[a, b] = cov_temp[(q * i) + a, (q * i) + b]

            cov_block = np.linalg.inv(cov_block)

            for a in range(q):
                for b in range(q):
                    cov_temp[(q * i) + a, (q * i) + b] = cov_block[a, b]

        s_MPL = np.dot(cov_temp, dx - dmut)
        print("Saving joint selection coefficients in %s/%s/%s as %s..." %(MPL_DIR, MPL_RAW_DIR, homolog, MPL_RAW_FILES[regular_list.index(regular)]))
        np.savetxt('%s/%s/%s/%s' % (MPL_DIR, MPL_RAW_DIR, homolog, MPL_RAW_FILES[regular_list.index(regular)]), s_MPL)



def single_replicate(**func_para):
 
    MPL_DIR      = func_para['MPL_DIR'] 
    DMS_DIR      = func_para['DMS_DIR']
    DATA_DIR     = func_para['DATA_DIR']
    PRE_FILE     = func_para['PRE_FILE']
    POST_FILE    = func_para['POST_FILE']
    MPL_RAW_DIR  = func_para['MPL_RAW_DIR']
    MPL_RAW_FILE = func_para['MPL_RAW_FILE']
    run_name     = func_para['run_name']
    homolog      = func_para['homolog']
    regular      = func_para['regular']
    err          = func_para['err']

    print("Calculating selection coefficients for replicate_%s of %s:" %(run_name,homolog))

    #df_0 = pd.read_csv('%s/data_%s/mutDNA-%s_codoncounts.csv'   % (DMS_DIR, homolog, run_name), comment = '#', memory_map = True)
    #df_1 = pd.read_csv('%s/data_%s/mutvirus-%s_codoncounts.csv' % (DMS_DIR, homolog, run_name), comment = '#', memory_map = True)
    df_0 = pd.read_csv('%s/%s/%s' % (DMS_DIR, DATA_DIR, PRE_FILE),  comment = '#', memory_map = True)
    df_1 = pd.read_csv('%s/%s/%s' % (DMS_DIR, DATA_DIR, POST_FILE), comment = '#', memory_map = True)
    q    = len(AA)
    L    = len(df_0)
    size = q * L
    x_0  = np.zeros(size)
    x_1  = np.zeros(size)
    dx   = np.zeros(size)
    dmut = np.zeros(size)
    cov  = np.zeros((size, size))

    wt = []
    for i in range(L):
        wt.append(str(df_0.iloc[i].wildtype))

    print("Correcting reads and computing allele frequency difference and mutational contribution...")
#new code
    norm_0 = np.zeros(L)
    norm_1 = np.zeros(L)
    for i in range(L):
        temp_1_wt = df_1.iloc[i][wt[i]]
        temp_0_wt = df_0.iloc[i][wt[i]]
        wt_index = CODONS.index(wt[i])
        
        x = df_0.iloc[i][CODONS].tolist() - temp_0_wt * err[i]
        y = df_1.iloc[i][CODONS].tolist() - temp_1_wt * err[i]
        
        x = [np.max([x_i,0]) for x_i in x]
        y = [np.max([y_i,0]) for y_i in y]

        x[wt_index] = temp_0_wt*err[i][wt_index]
        y[wt_index] = temp_1_wt*err[i][wt_index]
        norm_0[i] = sum(x)
        norm_1[i] = sum(y)


        x_0_c = [x_i/norm_0[i] for x_i in x]
        x_1_c = [x_i/norm_1[i] for x_i in y]
        
        
        for c in CODONS:    
            aa    = codon2aa(c)
            aaidx = AA.index(aa)
            
            x_0[(q * i) + aaidx] += x_0_c[CODONS.index(c)]
            x_1[(q * i) + aaidx] += x_1_c[CODONS.index(c)]
            dx[(q * i) + aaidx]  += x_1_c[CODONS.index(c)] - x_0_c[CODONS.index(c)]


            for c_i in range(3):
                for n in NUC:
                    if n!=c[c_i]:
                        m_aa    = codon2aa([c[k] if k !=c_i else n for k in range(3)])
                        m_aaidx = AA.index(m_aa)
                        dmut[(q * i) + aaidx]   -= x_0_c[CODONS.index(c)] * MU[c[c_i]+n]
                        dmut[(q * i) + m_aaidx] += x_0_c[CODONS.index(c)] * MU[c[c_i]+n]

#new code




    # Compute the total number of reads for each site
#    norm_0 = np.zeros(L)
#    norm_1 = np.zeros(L)
#    print("Correcting the total number of reads for each site with error correction...")
#    for c in CODONS:
#        for i in range(L):
#            if c==wt[i]:
#                norm_0[i] += df_0.iloc[i][c]*err[i][CODONS.index(c)]
#                norm_1[i] += df_1.iloc[i][c]*err[i][CODONS.index(c)]
#            else:
#                norm_0[i] += np.max([df_0.iloc[i][c] - df_0.iloc[i][wt[i]]*err[i][CODONS.index(c)], 0])
#                norm_1[i] += np.max([df_1.iloc[i][c] - df_1.iloc[i][wt[i]]*err[i][CODONS.index(c)], 0])

#    print("Computing allele frequency difference and mutational contribution...")

    # Compute dx and mutational contribution 
#    for c in CODONS:
#        aa    = codon2aa(c)
#        aaidx = AA.index(aa)
#        for i in range(L):

            # change in frequency
#            x_0_c = np.max([df_0.iloc[i][c] - df_0.iloc[i][wt[i]]*err[i][CODONS.index(c)], 0])
#            x_1_c = np.max([df_1.iloc[i][c] - df_1.iloc[i][wt[i]]*err[i][CODONS.index(c)], 0])
#            if c==wt[i]:
#                x_0_c = df_0.iloc[i][c]*err[i][CODONS.index(c)]
#                x_1_c = df_1.iloc[i][c]*err[i][CODONS.index(c)]
#            x_0_c /= norm_0[i]
#            x_1_c /= norm_1[i]

#            x_0[(q * i) + aaidx] += x_0_c
#            x_1[(q * i) + aaidx] += x_1_c
#            dx[(q * i) + aaidx]  += x_1_c - x_0_c

            # mutational flux
#            for c_i in range(3):
#                for n in NUC:
#                    if n!=c[c_i]:
#                        m_aa    = codon2aa([c[k] if k !=c_i else n for k in range(3)])
#                        m_aaidx = AA.index(m_aa)
#                        dmut[(q * i) + aaidx]   -= x_0_c * MU[c[c_i]+n]
#                        dmut[(q * i) + m_aaidx] += x_0_c * MU[c[c_i]+n]


    # Get wildtype sequence for amino acid
    wt = []
    for i in range(L):
        wt.append(AA.index(codon2aa(df_0.iloc[i].wildtype)))

    # Compute average frequencies for all states, and non-WT frequencies for each site
    x_avg = np.zeros(size)
    x_mut = np.zeros(L)
    for i in range(L):
        for a in range(q):
            x_avg[(q * i) + a] = (x_0[(q * i) + a] + x_1[(q * i) + a])/2

            if a!=wt[i]:
                x_mut[i] += x_avg[(q * i) + a]

    # Compute covariance
    for i in range(L):
        print("Computing covariance matrix...%.1f%%/100%% completed"%(float((i+1)/L)*100),end='\r')
        for a in range(q):

            # diagonal terms
            cov[(q * i) + a, (q * i) + a]  = x_avg[(q * i) + a] * (1 - x_avg[(q * i) + a])
            cov[(q * i) + a, (q * i) + a] += 1/regular  # diagonal regularization

            # off-diagonal, same site
            for b in range(a+1, q):
                cov[(q * i) + a, (q * i) + b] = -x_avg[(q * i) + a] * x_avg[(q * i) + b]
                cov[(q * i) + b, (q * i) + a] = -x_avg[(q * i) + a] * x_avg[(q * i) + b]

    cov_temp = cov.copy()
    for i in range(L):
        cov_block = np.zeros((q, q))
        for a in range(q):
            for b in range(q):
                cov_block[a, b] = cov_temp[(q * i) + a, (q * i) + b]

        cov_block = np.linalg.inv(cov_block)

        for a in range(q):
            for b in range(q):
                cov_temp[(q * i) + a, (q * i) + b] = cov_block[a, b]

    s_MPL = np.dot(cov_temp, dx - dmut)
    #s_MPL = np.dot(np.linalg.inv(cov), dx - dmut)
    #print("\tSaving selection coefficients in %s/raw_selection_coefficients/%s as %s_sMPL_%d.dat..."% (MPL_DIR, homolog, run_name, regular))
    #np.savetxt('%s/raw_selection_coefficients/%s/%s_sMPL_%d.dat' % (MPL_DIR, homolog, run_name, regular), s_MPL)
    print("Saving selection coefficients in %s/%s/%s as %s"% (MPL_DIR, MPL_RAW_DIR, homolog, MPL_RAW_FILE))
    np.savetxt('%s/%s/%s/%s' % (MPL_DIR, MPL_RAW_DIR, homolog, MPL_RAW_FILE), s_MPL)
    print("Replicate_%s of %s completed...\n"%(run_name, homolog))


def MPL_elements(**func_para):
 
    MPL_DIR      = func_para['MPL_DIR'] 
    DMS_DIR      = func_para['DMS_DIR']
    # DATA_DIR     = func_para['DATA_DIR']
    PRE_FILE     = func_para['PRE_FILE']
    POST_FILE    = func_para['POST_FILE']
    MPL_RAW_DIR  = func_para['MPL_RAW_DIR']
    # MPL_RAW_FILE = func_para['MPL_RAW_FILE']
    run_name     = func_para['run_name']
    homolog      = func_para['homolog']
    err          = func_para['err']

    print("Calculating selection coefficients for replicate_%s of %s:" %(run_name,homolog))

    #df_0 = pd.read_csv('%s/data_%s/mutDNA-%s_codoncounts.csv'   % (DMS_DIR, homolog, run_name), comment = '#', memory_map = True)
    #df_1 = pd.read_csv('%s/data_%s/mutvirus-%s_codoncounts.csv' % (DMS_DIR, homolog, run_name), comment = '#', memory_map = True)
    df_0 = pd.read_csv('%s/%s' % (DMS_DIR, PRE_FILE),  comment = '#', memory_map = True)
    df_1 = pd.read_csv('%s/%s' % (DMS_DIR, POST_FILE), comment = '#', memory_map = True)

    q    = len(AA)
    L    = len(df_0)
    size = q * L
    x_0  = np.zeros(size)
    x_1  = np.zeros(size)
    dx   = np.zeros(size)
    dmut = np.zeros(size)
    cov  = np.zeros((size, size))

    wt = []
    for i in range(L):
        wt.append(str(df_0.iloc[i].wildtype))

    print("Correcting reads and computing allele frequency difference and mutational contribution...")
#new code
    norm_0 = np.zeros(L)
    norm_1 = np.zeros(L)
    for i in range(L):
        temp_1_wt = df_1.iloc[i][wt[i]]
        temp_0_wt = df_0.iloc[i][wt[i]]
        wt_index = CODONS.index(wt[i])
        
        x = df_0.iloc[i][CODONS].tolist() - temp_0_wt * err[i]
        y = df_1.iloc[i][CODONS].tolist() - temp_1_wt * err[i]
        
        x = [np.max([x_i,0]) for x_i in x]
        y = [np.max([y_i,0]) for y_i in y]

        x[wt_index] = temp_0_wt*err[i][wt_index]
        y[wt_index] = temp_1_wt*err[i][wt_index]
        norm_0[i] = sum(x)
        norm_1[i] = sum(y)


        x_0_c = [x_i/norm_0[i] for x_i in x]
        x_1_c = [x_i/norm_1[i] for x_i in y]
        
        
        for c in CODONS:    
            aa    = codon2aa(c)
            aaidx = AA.index(aa)
            
            x_0[(q * i) + aaidx] += x_0_c[CODONS.index(c)]
            x_1[(q * i) + aaidx] += x_1_c[CODONS.index(c)]
            dx[(q * i) + aaidx]  += x_1_c[CODONS.index(c)] - x_0_c[CODONS.index(c)]


            for c_i in range(3):
                for n in NUC:
                    if n!=c[c_i]:
                        m_aa    = codon2aa([c[k] if k !=c_i else n for k in range(3)])
                        m_aaidx = AA.index(m_aa)
                        dmut[(q * i) + aaidx]   -= x_0_c[CODONS.index(c)] * MU[c[c_i]+n]
                        dmut[(q * i) + m_aaidx] += x_0_c[CODONS.index(c)] * MU[c[c_i]+n]

#new code


    max_read = np.max((norm_0.max(),norm_1.max()))
    # Compute the total number of reads for each site
#    norm_0 = np.zeros(L)
#    norm_1 = np.zeros(L)
#    print("Correcting the total number of reads for each site with error correction...")
#    for c in CODONS:
#        for i in range(L):
#            if c==wt[i]:
#                norm_0[i] += df_0.iloc[i][c]*err[i][CODONS.index(c)]
#                norm_1[i] += df_1.iloc[i][c]*err[i][CODONS.index(c)]
#            else:
#                norm_0[i] += np.max([df_0.iloc[i][c] - df_0.iloc[i][wt[i]]*err[i][CODONS.index(c)], 0])
#                norm_1[i] += np.max([df_1.iloc[i][c] - df_1.iloc[i][wt[i]]*err[i][CODONS.index(c)], 0])

#    print("Computing allele frequency difference and mutational contribution...")

    # Compute dx and mutational contribution 
#    for c in CODONS:
#        aa    = codon2aa(c)
#        aaidx = AA.index(aa)
#        for i in range(L):

            # change in frequency
#            x_0_c = np.max([df_0.iloc[i][c] - df_0.iloc[i][wt[i]]*err[i][CODONS.index(c)], 0])
#            x_1_c = np.max([df_1.iloc[i][c] - df_1.iloc[i][wt[i]]*err[i][CODONS.index(c)], 0])
#            if c==wt[i]:
#                x_0_c = df_0.iloc[i][c]*err[i][CODONS.index(c)]
#                x_1_c = df_1.iloc[i][c]*err[i][CODONS.index(c)]
#            x_0_c /= norm_0[i]
#            x_1_c /= norm_1[i]

#            x_0[(q * i) + aaidx] += x_0_c
#            x_1[(q * i) + aaidx] += x_1_c
#            dx[(q * i) + aaidx]  += x_1_c - x_0_c

            # mutational flux
#            for c_i in range(3):
#                for n in NUC:
#                    if n!=c[c_i]:
#                        m_aa    = codon2aa([c[k] if k !=c_i else n for k in range(3)])
#                        m_aaidx = AA.index(m_aa)
#                        dmut[(q * i) + aaidx]   -= x_0_c * MU[c[c_i]+n]
#                        dmut[(q * i) + m_aaidx] += x_0_c * MU[c[c_i]+n]


    # Get wildtype sequence for amino acid
    wt = []
    for i in range(L):
        wt.append(AA.index(codon2aa(df_0.iloc[i].wildtype)))

    # Compute average frequencies for all states, and non-WT frequencies for each site
    x_avg = np.zeros(size)
    x_mut = np.zeros(L)
    for i in range(L):
        for a in range(q):
            x_avg[(q * i) + a] = (x_0[(q * i) + a] + x_1[(q * i) + a])/2

            if a!=wt[i]:
                x_mut[i] += x_avg[(q * i) + a]

    # Compute covariance
    for i in range(L):
        # print("Computing covariance matrix...%.1f%%/100%% completed"%(float((i+1)/L)*100),end='\r')
        for a in range(q):

            # diagonal terms
            # cov[(q * i) + a, (q * i) + a]  = x_avg[(q * i) + a] * (1 - x_avg[(q * i) + a]) #old
            cov[(q * i) + a, (q * i) + a]  = (3 - 2 * x_1[(q * i) + a]) * (x_1[(q * i) + a] + x_0[(q * i) + a])/6 - (x_0[(q * i) + a] * x_0[(q * i) + a])/3
            # cov[(q * i) + a, (q * i) + a] += 1/regular  # diagonal regularization

            # off-diagonal, same site
            for b in range(a+1, q):
                # cov[(q * i) + a, (q * i) + b] = -x_avg[(q * i) + a] * x_avg[(q * i) + b] #old
                # cov[(q * i) + b, (q * i) + a] = -x_avg[(q * i) + a] * x_avg[(q * i) + b] #old
                cov[(q * i) + a, (q * i) + b] = -(2 * x_0[(q * i) + a] * x_0[(q * i) + b] + 2 * x_1[(q * i) + a] * x_1[(q * i) + b] + x_1[(q * i) + a] * x_0[(q * i) + b] + x_1[(q * i) + b] * x_0[(q * i) + a])/6
                cov[(q * i) + b, (q * i) + a] = -(2 * x_0[(q * i) + a] * x_0[(q * i) + b] + 2 * x_1[(q * i) + a] * x_1[(q * i) + b] + x_1[(q * i) + a] * x_0[(q * i) + b] + x_1[(q * i) + b] * x_0[(q * i) + a])/6
    cov_temp = cov.copy()
    return dx - dmut, cov_temp, L, q, max_read


    # for i in range(L):
    #     cov_block = np.zeros((q, q))
    #     for a in range(q):
    #         for b in range(q):
    #             cov_block[a, b] = cov_temp[(q * i) + a, (q * i) + b]

    #     cov_block = np.linalg.inv(cov_block)

    #     for a in range(q):
    #         for b in range(q):
    #             cov_temp[(q * i) + a, (q * i) + b] = cov_block[a, b]

    # s_MPL = np.dot(cov_temp, dx - dmut)
    # #s_MPL = np.dot(np.linalg.inv(cov), dx - dmut)
    # #print("\tSaving selection coefficients in %s/raw_selection_coefficients/%s as %s_sMPL_%d.dat..."% (MPL_DIR, homolog, run_name, regular))
    # #np.savetxt('%s/raw_selection_coefficients/%s/%s_sMPL_%d.dat' % (MPL_DIR, homolog, run_name, regular), s_MPL)
    # print("Saving selection coefficients in %s/%s/%s as %s"% (MPL_DIR, MPL_RAW_DIR, homolog, MPL_RAW_FILE))
    # np.savetxt('%s/%s/%s/%s' % (MPL_DIR, MPL_RAW_DIR, homolog, MPL_RAW_FILE), s_MPL)
    # print("Replicate_%s of %s completed...\n"%(run_name, homolog))


def enrichment_ratio(**func_para):
 
    MPL_DIR      = func_para['MPL_DIR'] 
    DMS_DIR      = func_para['DMS_DIR']
    # DATA_DIR     = func_para['DATA_DIR']
    PRE_FILE     = func_para['PRE_FILE']
    POST_FILE    = func_para['POST_FILE']
    MPL_RAW_DIR  = func_para['MPL_RAW_DIR']
    # MPL_RAW_FILE = func_para['MPL_RAW_FILE']
    run_name     = func_para['run_name']
    homolog      = func_para['homolog']
    err          = func_para['err']

    print("Calculating selection coefficients for replicate_%s of %s:" %(run_name,homolog))

    #df_0 = pd.read_csv('%s/data_%s/mutDNA-%s_codoncounts.csv'   % (DMS_DIR, homolog, run_name), comment = '#', memory_map = True)
    #df_1 = pd.read_csv('%s/data_%s/mutvirus-%s_codoncounts.csv' % (DMS_DIR, homolog, run_name), comment = '#', memory_map = True)
    df_0 = pd.read_csv('%s/%s' % (DMS_DIR, PRE_FILE),  comment = '#', memory_map = True)
    df_1 = pd.read_csv('%s/%s' % (DMS_DIR, POST_FILE), comment = '#', memory_map = True)

    q    = len(AA)
    L    = len(df_0)
    size = q * L
    x_0  = np.zeros(size)
    x_1  = np.zeros(size)
    enrich_ratio = np.zeros(size)
    # dx_0   = np.zeros(size)
    # dx_1   = np.zeros(size)
    dmut = np.zeros(size)
    cov  = np.zeros((size, size))

    wt = []
    for i in range(L):
        wt.append(str(df_0.iloc[i].wildtype))

    print("Correcting reads and computing allele frequency difference and mutational contribution...")
#new code
    norm_0 = np.zeros(L)
    norm_1 = np.zeros(L)
    for i in range(L):
        temp_1_wt = df_1.iloc[i][wt[i]]
        temp_0_wt = df_0.iloc[i][wt[i]]
        wt_index = CODONS.index(wt[i])
        
        x = df_0.iloc[i][CODONS].tolist() - temp_0_wt * err[i]
        y = df_1.iloc[i][CODONS].tolist() - temp_1_wt * err[i]
        
        x = [np.max([x_i,0]) for x_i in x]
        y = [np.max([y_i,0]) for y_i in y]

        x[wt_index] = temp_0_wt*err[i][wt_index]
        y[wt_index] = temp_1_wt*err[i][wt_index]
        norm_0[i] = sum(x)
        norm_1[i] = sum(y)

        x_0_c = [x_i/norm_0[i] for x_i in x]
        x_1_c = [x_i/norm_1[i] for x_i in y]
        
        
        for c in CODONS:    
            aa    = codon2aa(c)
            aaidx = AA.index(aa)
            x_0[(q * i) + aaidx] += x_0_c[CODONS.index(c)]
            x_1[(q * i) + aaidx] += x_1_c[CODONS.index(c)]

    for i in range(len(x_1)):
        if x_0[i] != 0:
            enrich_ratio[i] += x_1[i]/x_0[i]
        else:
            enrich_ratio[i] += -1

    return enrich_ratio




def checksum(**func_para):

    MPL_DIR       = func_para['MPL_DIR']
    MPL_RAW_DIR   = func_para['MPL_RAW_DIR']
    MPL_RAW_FILES = func_para['MPL_RAW_FILES']
    homolog       = func_para['homolog']
    regular_list  = func_para['regular_list']
    runname_index = func_para['runname_index']
    threshold     = func_para['threshold']

    print('--- Check summation of selection coefficients on each site, and the expectation should be close to 0---\n')
    print('%s test (threshold=%.1e, R is regularization term):' % (homolog, threshold))
    for regular in regular_list:
        s_i = np.loadtxt('%s/%s/%s/%s' %(MPL_DIR, MPL_RAW_DIR, homolog, MPL_RAW_FILES[regular_list.index(regular)]))
        file_length = len(s_i)
        sequence_length = int(file_length/21)
        counts = 0

        for site in range(sequence_length):
            if np.sum(s_i[site * 21: (site + 1) * 21]) < threshold:
                counts += 1

        if runname_index == 'joint':
            print("Joint selection coefficients (R = %d) summation on (%d out of %d) sites are close to 0" % (regular, counts, sequence_length))
        else:
            for run_i in runname_index:
                print("Replicate %s selection coefficients (R = %d) summation on (%d out of %d) sites are close to 0" % (run_i, regular, counts, sequence_length))
    print('\n')



def correlation_plot(**func_para):

    MPL_DIR       = func_para['MPL_DIR']
    DMS_DIR       = func_para['DMS_DIR']
    PREFS_DIR     = func_para['PREFS_DIR']
    RES_FILE      = func_para['RES_FILE']
    MPL_RAW_DIR   = func_para['MPL_RAW_DIR']
    MPL_RAW_FILES = func_para['MPL_RAW_FILES']
    homolog       = func_para['homolog']
    regular_list  = func_para['regular_list']

    s_i = pd.read_csv('%s/%s/%s' %(DMS_DIR, PREFS_DIR, RES_FILE), comment = '#', memory_map = True)
    #DMS/prefs/rescaled-...
    v_i = []
    col = list(s_i.columns)
    col.remove('site')
    for ii in range(len(s_i)):
        for c in col:
            v_i.append(float(s_i.iloc[ii][c]))
    v_i = np.array(v_i)

    pearson_value = []
    spearman_value = []

    for regular in regular_list:
        s_j = []
        #s_MPL = np.loadtxt('MPL/raw_selection_coefficients/BG505/joint_sMPL_1.dat')
        s_MPL = np.loadtxt('%s/%s/%s/%s' %(MPL_DIR, MPL_RAW_DIR, homolog, MPL_RAW_FILES[regular_list.index(regular)]))
        s_j = np.delete(s_MPL, np.arange(20, len(s_MPL), 21))

        pearson_value.append(st.pearsonr(v_i, s_j)[0])
        spearman_value.append(st.spearmanr(v_i, s_j)[0])

    plot.scatter(regular_list, pearson_value, alpha = 0.5, label = '%s: Pearson' %homolog)
    regular_list_new = np.linspace(min(regular_list), max(regular_list), 1000)
    a_BSpline = interpolate.make_interp_spline(regular_list, pearson_value)
    pearson_value_new = a_BSpline(regular_list_new)
    plot.plot(regular_list_new, pearson_value_new, alpha = 0.5)

    plot.scatter(regular_list, spearman_value, alpha = 0.5, label = '%s: Spearman' %homolog)
    regular_list_new = np.linspace(min(regular_list), max(regular_list), 1000)
    a_BSpline = interpolate.make_interp_spline(regular_list, spearman_value)
    spearman_value_new = a_BSpline(regular_list_new)
    plot.plot(regular_list_new, spearman_value_new, alpha = 0.5)



def data_merge(**func_para):

    file_columns =  func_para['file_columns']
    replicates   =  func_para['replicates']
    MPL_DIR      =  func_para['MPL_DIR']
    DMS_DIR      =  func_para['DMS_DIR']
    PREFS_DIR    =  func_para['PREFS_DIR']
    MPL_RAW_DIR  =  func_para['MPL_RAW_DIR']
    homolog      =  func_para['homolog']
    beta         =  func_para['beta']
    regular      =  func_para['regular']
    COM_DIR      =  func_para['COM_DIR']
    COM_FILE     =  func_para['COM_FILE']

    decimal_digits = 6
    av_i_res       = []
    av_i           = []
    as_j_nor       = []
    as_j           = []
    aamino_acid_list = []
    areplicate_list  = []
    asite_list       = []



    for replicate in replicates:
        v_i_res = []
        v_i     = []
        s_j_nor = []
        s_j     = []
        amino_acid_list = []
        replicate_list  = []
        site_list       = []


        if replicate != 'Average/Joint':
            s_i_res = pd.read_csv('%s/%s/rescaled_%s-%s_prefs.csv' %(DMS_DIR, PREFS_DIR, homolog, replicate), comment = '#', memory_map = True)
            s_i = pd.read_csv('%s/%s/%s-%s_prefs.csv' %(DMS_DIR, PREFS_DIR, homolog, replicate), comment = '#', memory_map = True)
            s_MPL = np.loadtxt('%s/%s/%s/%s_sMPL_%d.dat' %(MPL_DIR, MPL_RAW_DIR, homolog, replicate, regular))

        else:
            s_i_res = pd.read_csv('%s/%s/rescaled_%s_avgprefs.csv' %(DMS_DIR, PREFS_DIR, homolog), comment = '#', memory_map = True)
            s_i = pd.read_csv('%s/%s/%s_avgprefs.csv' %(DMS_DIR, PREFS_DIR, homolog), comment = '#', memory_map = True)
            s_MPL = np.loadtxt('%s/%s/%s/joint_sMPL_%d.dat'%(MPL_DIR, MPL_RAW_DIR, homolog, regular))
        
        col = list(s_i.columns)
        site_index = s_i_res['site'].tolist()
        col.remove('site')
        for ii in range(len(s_i_res)):
            for c in col:
                v_i_res.append(float(s_i_res.iloc[ii][c]))
                v_i.append(float(s_i.iloc[ii][c]))
                replicate_list.append(replicate)
                amino_acid_list.append(c)
                site_list.append(site_index[ii])
        v_i_res = np.array(v_i_res)
        v_i = np.array(v_i)
        s_j = np.append(s_j, np.delete(s_MPL, np.arange(20, len(s_MPL), 21)))

        for i in range(int(len(s_j)/20)):
            summation=0
            for j in range(20):
                summation+=math.exp(s_j[i*20+j]*beta)
            for j in range(20):
                s_j_nor.append(math.exp(s_j[i*20+j]*beta)/summation)  

                    
        av_i_res = np.append(av_i_res, v_i_res)
        av_i     = np.append(av_i, v_i)
        as_j_nor = np.append(as_j_nor, s_j_nor)
        as_j  = np.append(as_j, s_j)
        aamino_acid_list  = np.append(aamino_acid_list, amino_acid_list)
        areplicate_list   = np.append(areplicate_list, replicate_list)
        asite_list  = np.append(asite_list, site_list)


    #print(beta)
    av_i_res = [round(val, decimal_digits) for val in av_i_res]
    av_i  = [round(val, decimal_digits) for val in av_i]
    as_j_nor = [round(val, decimal_digits) for val in as_j_nor]
    as_j  = [round(val, decimal_digits) for val in as_j]


    with open('%s/%s.csv' %(COM_DIR, COM_FILE) , mode = 'w') as file:

        file_writer = csv.writer(file, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
        file_writer.writerow(file_columns)
        
        for i in range(len(asite_list)):
            file_writer.writerow([areplicate_list[i], asite_list[i], aamino_acid_list[i],
                                av_i[i], av_i_res[i], as_j[i], as_j_nor[i]])
        file.close()    



def corrfunc_heat(x, y, **kws):
    r, _ = st.pearsonr(x, y)
    ax   = plot.gca()
    ax.annotate("R = {:.2f}".format(r),
                xy=(.1, .9), xycoords = ax.transAxes)



def pairgrid_scatter(**func_para):
    method        = func_para['method']
    homologs      = func_para['homologs']
    runname_index = func_para['runname_index']
    df_dmswithmpl = func_para['df_dmswithmpl']
    alpha         = func_para['alpha']
    s             = func_para['dot_size']
    color         = func_para['color']
    xticks        = func_para['xticks']
    yticks        = func_para['yticks']
    xlim          = func_para['xlim']
    ylim          = func_para['ylim']

    pairgrid_data = {}
    pairgrid_key  = []

    for homolog in homologs:
        for replicate in runname_index:
            key_name = homolog + '-' + replicate
            pairgrid_key.append(key_name)
            col_name = method + '_' + homolog
            df_prefs_temp = df_dmswithmpl.loc[df_dmswithmpl['replicate'] == replicate, col_name].tolist()
            pairgrid_data.update({key_name: df_prefs_temp})

    df_prefs_pairgrid = pd.DataFrame(data = pairgrid_data)

    g = sns.PairGrid(df_prefs_pairgrid, vars = pairgrid_key)
    g.map(plot.scatter, alpha = alpha, s = s, linewidth = 0, edgecolors = 'None', color = color)
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        g.axes[i, j].set_visible(False)
    for i, j in zip(*np.triu_indices_from(g.axes, 0)):
        g.axes[i, i].set_visible(False)
    g.set(xticks = xticks, yticks = yticks, xlim = xlim, ylim = ylim)
    g.map_lower(corrfunc_heat)



def pairgrid_heatmap(**func_para):

    homologs      = func_para['homologs']
    runname_index = func_para['runname_index']
    df_dmswithmpl = func_para['df_dmswithmpl']
    color         = func_para['color']
    space         = func_para['space']
    shrink        = func_para['shrink']

    pairgrid_data = {}
    pairgrid_key  = []
    for homolog in homologs:
        for replicate in runname_index:
            key_name = homolog + '-' + replicate
            pairgrid_key.append(key_name)
            col_name = 'preference_' + homolog
            df_prefs_temp = df_dmswithmpl.loc[df_dmswithmpl['replicate'] == replicate, col_name].tolist()
            pairgrid_data.update({key_name: df_prefs_temp})

    df_prefs_pairgrid = pd.DataFrame(data = pairgrid_data)
    matrix_0 = np.tril(df_prefs_pairgrid.corr())

    pairgrid_data = {}
    pairgrid_key = []
    for homolog in homologs:
        for replicate in runname_index:
            key_name = homolog + '-' + replicate
            pairgrid_key.append(key_name)
            col_name = 'selection_coefficient_' + homolog
            df_sele_temp = df_dmswithmpl.loc[df_dmswithmpl['replicate'] == replicate, col_name].tolist()
            pairgrid_data.update({key_name: df_sele_temp})

    df_sele_temp = pd.DataFrame(data = pairgrid_data)
    matrix_1 = np.triu(df_sele_temp.corr())

    plot.figure(figsize = (8, 8))
    sns.heatmap(df_sele_temp.corr(),      annot = True, vmin = 0.35, vmax = 1, square = True, mask = matrix_0, cmap = color, cbar_kws = {"shrink": shrink}, linewidths = space)
    sns.heatmap(df_prefs_pairgrid.corr(), annot = True, vmin = 0.35, vmax = 1, square = True, mask = matrix_1, cmap = color, cbar = False, linewidths = space)
    plot.yticks(rotation = 0)
    plot.xticks(rotation = 0)
    plot.show()



def correlation_hist(**func_para):

    PREFS_COL  = func_para['PREFS_COL']
    SELEC_COL  = func_para['SELEC_COL']
    df         = func_para['df']
    bins       = func_para['bins']
    alpha      = func_para['alpha']
    homolog    = func_para['homolog']
    legend_pos = func_para['legend_pos']

    pearson_values  = []
    spearman_values = []
 

    preference_list = df.loc[df['replicate'] == 'Average/Joint', '%s' %PREFS_COL].tolist()
    selection_coefficient_list = df.loc[df['replicate'] == 'Average/Joint', '%s' %SELEC_COL].tolist()
    preference_list = [x for x in preference_list if ~np.isnan(x)]
    selection_coefficient_list = [x for x in selection_coefficient_list if ~np.isnan(x)]

    for i in range(int(len(preference_list)/20)):
        pearson_values.append(st.pearsonr(preference_list[i * 20: i * 20 + 19], selection_coefficient_list[i * 20: i * 20 + 19])[0])
        spearman_values.append(st.spearmanr(preference_list[i * 20: i * 20 + 19], selection_coefficient_list[i * 20: i * 20 + 19])[0])
  
    low_lim = np.min([np.min(spearman_values), np.min(pearson_values)]) - 0.1
    plot.hist(pearson_values,  bins = bins, alpha = alpha, label = 'Pearson',  range = [low_lim, 1], histtype = u'step', fill = True)
    plot.hist(spearman_values, bins = bins, alpha = alpha, label = 'Spearman', range = [low_lim, 1], histtype = u'step', fill = True)
    plot.legend(loc = '%s' %legend_pos, fontsize = 13)
    plot.title('%s' %homolog)
    plot.show()













