import sys, os
from importlib import reload
import copy
import itertools
from itertools import combinations
import numpy as np
import itertools
import scipy as sp
import scipy.stats as st
from scipy import interpolate 

import pandas as pd
import seaborn as sns

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

import tqdm
import time
import csv
import math

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
AA  = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I','L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '*']
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


codon_to_aa_num = {}
for i in range(len(CODONS)):
    codon = CODONS[i]
    aminoacid = codon2aa(codon)
    aminoacid_num = AA.index(aminoacid)
    codon_to_aa_num[i] = aminoacid_num


# codes for short read data

def short_read_pipeline(DNACODON, TARGET_PROTEIN, MPL_DIR, MPL_RAW_DIR, REPLICATES, SITE_START, SITE_END, INPUT_DIR, OUTPUT_DIR, EPISTASIS, REGULARIZATION_PERCENT):
    estimate_selection, regularization_list = MPL_short_read_inference(TARGET_PROTEIN, DNACODON, REPLICATES, MPL_DIR, MPL_RAW_DIR, INPUT_DIR)
    correlation_list = optimize_regularization_short_read(estimate_selection, REPLICATES, regularization_list)
    plot_reg_corr(regularization_list, correlation_list, TARGET_PROTEIN)
    REGULARIZATION_SELECTED = find_best_regularization(regularization_list, correlation_list, REGULARIZATION_PERCENT)
    FINAL_SELECTION = output_final_selection_short_read(REGULARIZATION_SELECTED, SITE_END, SITE_START, estimate_selection, REPLICATES)
    
    if os.path.exists(OUTPUT_DIR):
        FINAL_SELECTION.to_csv(OUTPUT_DIR+TARGET_PROTEIN+'_'+'%d.csv.gz' %int(np.log10(REGULARIZATION_SELECTED)), index = False, compression = 'gzip')
    else:
        os.makedirs(OUTPUT_DIR)
        FINAL_SELECTION.to_csv(OUTPUT_DIR+TARGET_PROTEIN+'_'+'%d.csv.gz' %int(np.log10(REGULARIZATION_SELECTED)), index = False, compression = 'gzip')

def optimize_regularization_short_read(estimate_selection, REPLICATES, REGULARIZATION_LIST):
    correlation_list = []
    for regular in REGULARIZATION_LIST:
        temp_sele=[]
        temp_corr = []
        for replicate in REPLICATES:
            temp_sele.append(estimate_selection[str(replicate)][regular])
        for pairs in list(combinations(temp_sele, 2)):
            temp_corr.append(st.pearsonr(pairs[0], pairs[1])[0])
        correlation_list.append(np.mean(temp_corr))
    return correlation_list

def plot_reg_corr(regularization_list, correlation_list, protein_name):
    plt.plot(regularization_list, correlation_list)
    plt.xscale('log')
    plt.xlabel('Regularization')
    plt.ylabel('Pearson Correlation')
    plt.title(protein_name+' regularization plot')
    plt.show()

def find_best_regularization(reg_list, corr_list, percent):
    threshold = (max(corr_list)-min(corr_list))/percent
    max_index = corr_list.index(max(corr_list))
    if abs(corr_list[max_index]-corr_list[0])<0.01:
        optimized_regularization = reg_list[0]
    else:
        for i in range(max_index,0,-1):
            if abs(corr_list[i]-corr_list[i-1])>=threshold*corr_list[i]:
                optimized_regularization = reg_list[i]
                break  
    print('Optimized regularization term = %.1e' %optimized_regularization)
    return optimized_regularization


def cov_inverse(cov, L, q, regular):
    cov_temp = cov.copy()
    for i in range(L):
        for a in range(q):
            cov_temp[(q * i) + a, (q * i) + a] += regular  # diagonal regularization
    
    for i in range(L):
        cov_block = np.zeros((q, q))
        for a in range(q):
            for b in range(q):
                cov_block[a, b] = cov_temp[(q * i) + a, (q * i) + b]

        cov_block = np.linalg.inv(cov_block)

        for a in range(q):
            for b in range(q):
                cov_temp[(q * i) + a, (q * i) + b] = cov_block[a, b]

    return cov_temp

def output_final_selection_short_read(REGULARIZATION_SELECTED, SITE_END, SITE_START, estimate_selection, REPLICATES):
    site_list = []
    for site in range(SITE_END-SITE_START+1):
        site_list+=[site]*21
    AA_list = AA*(SITE_END-SITE_START+1)
    rep_list = []
    for replicate in REPLICATES:
        rep_list.append('rep_'+str(replicate))
    df_selection = pd.DataFrame(columns=['site', 'amino_acid']+rep_list)
    df_selection['site'] = site_list
    df_selection['amino_acid'] = AA_list
    for replicate in REPLICATES:
        df_selection['rep_'+str(replicate)] = estimate_selection[str(replicate)][REGULARIZATION_SELECTED]
    df_selection['joint'] = estimate_selection['joint'][REGULARIZATION_SELECTED]
    return df_selection   
        
def MPL_short_read_inference(TARGET_PROTEIN, DNACODON, REPLICATES, MPL_DIR, MPL_RAW_DIR, INPUT_DIR):
    print('------ Calculating single replicate selection coefficients for %s ------' %TARGET_PROTEIN)
    print("\nCalculating error probability from %s...\n" %DNACODON)

    err = err_correct(INPUT_DIR, DNACODON)
    estimate_selection = {}
    
    minimum = sys.float_info.max
    

    for run_name in REPLICATES:
        run_name = str(run_name)
        estimate_selection[run_name]={}
        func_para = {
            'MPL_DIR':      MPL_DIR,                                             # MPL directory
            'DMS_DIR':      INPUT_DIR,                                             # DMS directory
            'PRE_FILE':     'mutDNA-' + run_name + '_codoncounts.csv',           # Pre-count data file
            'POST_FILE':    'mutvirus-' + run_name + '_codoncounts.csv',         # Post-count data filr
            'MPL_RAW_DIR':  MPL_RAW_DIR,                                         # MPL raw selection coefficients directory
            'run_name':     run_name,                                            # Replicate serial number
            'homolog':      TARGET_PROTEIN,                                      # Homolog name
            'err':          err                                                  # Error probability for this homolog
        }
        
        FREQUENCY_DIFF, COVARIANCE_MATRIX, L, q, max_read = MPL_short_read_elements(**func_para)
        if run_name == str(REPLICATES[0]):
            joint_freq_diff = np.array(FREQUENCY_DIFF)
            joint_cov_mat = np.array(COVARIANCE_MATRIX)
        else:
            joint_freq_diff = np.add(joint_freq_diff, FREQUENCY_DIFF)
            joint_cov_mat = np.add(joint_cov_mat, COVARIANCE_MATRIX)
        if minimum > 1e50:
            minimum = max_read
            regularization_list = [np.power(10, float(i)) for i in range(int(np.log10(1/minimum)-1), 4)]
        for REGULAR in regularization_list:
            INVERSE_COV = cov_inverse(COVARIANCE_MATRIX, L, q, REGULAR)
            estimate_selection[run_name][REGULAR] = np.dot(INVERSE_COV, FREQUENCY_DIFF)
    estimate_selection['joint']={}
    for REGULAR in regularization_list:
            INVERSE_COV = cov_inverse(joint_cov_mat, L, q, REGULAR)
            estimate_selection['joint'][REGULAR] = np.dot(INVERSE_COV, joint_freq_diff)        
    
    return estimate_selection, regularization_list



def err_correct(DMS_DIR, DNACODON):

    df_0 = pd.read_csv('%s%s' % (DMS_DIR, DNACODON), comment = '#', memory_map = True)     # Read raw data file

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

    # Compute error probability
    for i in range(L):
        err[i] = df_0.iloc[i][CODONS].tolist()/norm_0[i]       
        wild_type_index = CODONS.index(wt[i])       
        err[i][wild_type_index] = norm_0[i]/df_0.iloc[i][CODONS].tolist()[wild_type_index]

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

    df_0 = pd.read_csv('%s/%s%s' % (DMS_DIR, DATA_DIR, PRE_FILES[0]), comment = '#', memory_map = True)
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
        df_0 = pd.read_csv('%s/%s%s' % (DMS_DIR, DATA_DIR, PRE_FILES[index]),  comment = '#', memory_map = True)
        df_1 = pd.read_csv('%s/%s%s' % (DMS_DIR, DATA_DIR, POST_FILES[index]), comment = '#', memory_map = True)

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

    df_0 = pd.read_csv('%s/%s%s' % (DMS_DIR, DATA_DIR, PRE_FILE),  comment = '#', memory_map = True)
    df_1 = pd.read_csv('%s/%s%s' % (DMS_DIR, DATA_DIR, POST_FILE), comment = '#', memory_map = True)
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
    print("Saving selection coefficients in %s/%s/%s as %s"% (MPL_DIR, MPL_RAW_DIR, homolog, MPL_RAW_FILE))
    np.savetxt('%s/%s/%s/%s' % (MPL_DIR, MPL_RAW_DIR, homolog, MPL_RAW_FILE), s_MPL)
    print("Replicate_%s of %s completed...\n"%(run_name, homolog))


def MPL_short_read_elements(**func_para):
 
    MPL_DIR      = func_para['MPL_DIR'] 
    DMS_DIR      = func_para['DMS_DIR']
    PRE_FILE     = func_para['PRE_FILE']
    POST_FILE    = func_para['POST_FILE']
    MPL_RAW_DIR  = func_para['MPL_RAW_DIR']
    run_name     = func_para['run_name']
    homolog      = func_para['homolog']
    err          = func_para['err']

    print("Calculating selection coefficients for replicate_%s of %s:" %(run_name,homolog))

    df_0 = pd.read_csv('%s%s' % (DMS_DIR, PRE_FILE),  comment = '#', memory_map = True)
    df_1 = pd.read_csv('%s%s' % (DMS_DIR, POST_FILE), comment = '#', memory_map = True)

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

    max_read = np.max((norm_0.max(),norm_1.max()))

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
            cov[(q * i) + a, (q * i) + a]  = (3 - 2 * x_1[(q * i) + a]) * (x_1[(q * i) + a] + x_0[(q * i) + a])/6 - (x_0[(q * i) + a] * x_0[(q * i) + a])/3

            # off-diagonal, same site
            for b in range(a+1, q):
                cov[(q * i) + a, (q * i) + b] = -(2 * x_0[(q * i) + a] * x_0[(q * i) + b] + 2 * x_1[(q * i) + a] * x_1[(q * i) + b] + x_1[(q * i) + a] * x_0[(q * i) + b] + x_1[(q * i) + b] * x_0[(q * i) + a])/6
                cov[(q * i) + b, (q * i) + a] = -(2 * x_0[(q * i) + a] * x_0[(q * i) + b] + 2 * x_1[(q * i) + a] * x_1[(q * i) + b] + x_1[(q * i) + a] * x_0[(q * i) + b] + x_1[(q * i) + b] * x_0[(q * i) + a])/6
    cov_temp = cov.copy()
    return dx - dmut, cov_temp, L, q, max_read

def enrichment_ratio(**func_para):
 
    MPL_DIR      = func_para['MPL_DIR'] 
    DMS_DIR      = func_para['DMS_DIR']
    # DATA_DIR     = func_para['DATA_DIR']
    PRE_FILE     = func_para['PRE_FILE']
    POST_FILE    = func_para['POST_FILE']
    MPL_RAW_DIR  = func_para['MPL_RAW_DIR']
    run_name     = func_para['run_name']
    homolog      = func_para['homolog']
    err          = func_para['err']

    print("Calculating selection coefficients for replicate_%s of %s:" %(run_name,homolog))

    df_0 = pd.read_csv('%s%s' % (DMS_DIR, PRE_FILE),  comment = '#', memory_map = True)
    df_1 = pd.read_csv('%s%s' % (DMS_DIR, POST_FILE), comment = '#', memory_map = True)

    q    = len(AA)
    L    = len(df_0)
    size = q * L
    x_0  = np.zeros(size)
    x_1  = np.zeros(size)
    enrich_ratio = np.zeros(size)
    dmut = np.zeros(size)
    cov  = np.zeros((size, size))

    wt = []
    for i in range(L):
        wt.append(str(df_0.iloc[i].wildtype))

    print("Correcting reads and computing allele frequency difference and mutational contribution...")
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

    plt.scatter(regular_list, pearson_value, alpha = 0.5, label = '%s: Pearson' %homolog)
    regular_list_new = np.linspace(min(regular_list), max(regular_list), 1000)
    a_BSpline = interpolate.make_interp_spline(regular_list, pearson_value)
    pearson_value_new = a_BSpline(regular_list_new)
    plt.plot(regular_list_new, pearson_value_new, alpha = 0.5)

    plt.scatter(regular_list, spearman_value, alpha = 0.5, label = '%s: Spearman' %homolog)
    regular_list_new = np.linspace(min(regular_list), max(regular_list), 1000)
    a_BSpline = interpolate.make_interp_spline(regular_list, spearman_value)
    spearman_value_new = a_BSpline(regular_list_new)
    plt.plot(regular_list_new, spearman_value_new, alpha = 0.5)



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
    ax   = plt.gca()
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
    g.map(plt.scatter, alpha = alpha, s = s, linewidth = 0, edgecolors = 'None', color = color)
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

    plt.figure(figsize = (8, 8))
    sns.heatmap(df_sele_temp.corr(),      annot = True, vmin = 0.35, vmax = 1, square = True, mask = matrix_0, cmap = color, cbar_kws = {"shrink": shrink}, linewidths = space)
    sns.heatmap(df_prefs_pairgrid.corr(), annot = True, vmin = 0.35, vmax = 1, square = True, mask = matrix_1, cmap = color, cbar = False, linewidths = space)
    plt.yticks(rotation = 0)
    plt.xticks(rotation = 0)
    plt.show()

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
    plt.hist(pearson_values,  bins = bins, alpha = alpha, label = 'Pearson',  range = [low_lim, 1], histtype = u'step', fill = True)
    plt.hist(spearman_values, bins = bins, alpha = alpha, label = 'Spearman', range = [low_lim, 1], histtype = u'step', fill = True)
    plt.legend(loc = '%s' %legend_pos, fontsize = 13)
    plt.title('%s' %homolog)
    plt.show()


# codes for full genome data

def full_genome_pipeline(TARGET_PROTEIN, REPLICATES, SITE_START, SITE_END, INPUT_DIR, OUTPUT_DIR, EPISTASIS, REGULARIZATION_PERCENT):
    INPUT_FILE_PREFIX, FLAG_LIST, SITES = initialization(SITE_START, SITE_END, INPUT_DIR, OUTPUT_DIR, EPISTASIS)

    FREQUENCY_DIFF, COVARIANCE_MATRIX, MAX_READ = MPL_full_genome_elements(FLAG_LIST, INPUT_DIR, REPLICATES, INPUT_FILE_PREFIX, SITES)
    REGULARIZATION_LIST = [np.power(10, float(i)) for i in range(int(np.log10(1/MAX_READ)-1), 4)]
    CORRELATION_LIST = optimize_regularization_full_genome(REGULARIZATION_LIST, SITES, COVARIANCE_MATRIX, FREQUENCY_DIFF, REPLICATES)

    plot_reg_corr(REGULARIZATION_LIST, CORRELATION_LIST, TARGET_PROTEIN)

    REGULARIZATION_SELECTED = find_best_regularization(REGULARIZATION_LIST, CORRELATION_LIST, REGULARIZATION_PERCENT)

    FINAL_SELECTION = output_final_selection_full_genome(REGULARIZATION_SELECTED, SITES, COVARIANCE_MATRIX, FREQUENCY_DIFF, REPLICATES)
    
    if os.path.exists(OUTPUT_DIR):
        FINAL_SELECTION.to_csv(OUTPUT_DIR+TARGET_PROTEIN+'_'+'%d.csv.gz' %int(np.log10(REGULARIZATION_SELECTED)), index = False, compression = 'gzip')
    else:
        os.makedirs(OUTPUT_DIR)
        FINAL_SELECTION.to_csv(OUTPUT_DIR+TARGET_PROTEIN+'_'+'%d.csv.gz' %int(np.log10(REGULARIZATION_SELECTED)), index = False, compression = 'gzip')

def MPL_full_genome_elements(flag_list, Input_dir, replicates, Input_file_prefix, sites):
    df_counts_dict = {}
    max_read_final = 0

    for flag in flag_list:
        print('Reading %s allele counts files from:' %flag)
        df_counts_dict[flag] = get_counts(Input_dir, replicates, Input_file_prefix, flag)

    df_frequency_dict = {}

    for flag in flag_list:
        df_frequency_dict[flag], max_read = counts_to_frequency(df_counts_dict, replicates, flag, sites)
        max_read_final = max(max_read_final, max_read)

    df_frequency_change_dict = {}

    for flag in flag_list:
        df_frequency_change_dict[flag] = frequency_change(df_frequency_dict, flag)

    cov_matx = covariance_matrix(df_frequency_dict, flag_list, sites)
    
    return df_frequency_change_dict, cov_matx, max_read_final

def output_final_selection_full_genome(gamma, sites, cov_matx, df_frequency_change_dict, replicates):
    copy_cov_matx = copy.deepcopy(cov_matx['amino_acid'])
    q = len(AA)
    L = len(sites)
    estimate_selection = {}
    
    joint_freq_change = np.array([0]*(q*L))
    joint_cov_mat = np.array([[0]*(q*L)]*(q*L))

    for rep in list(copy_cov_matx.keys()):
        invert_matrix = np.zeros((q * L, q * L))

        for i in range(len(list(copy_cov_matx[rep].keys())) - 1):
            time_i = list(copy_cov_matx[rep].keys())[i]
            time_i_post = list(copy_cov_matx[rep].keys())[i+1]
            time_interval = (time_i_post- time_i)
            invert_matrix += time_interval * copy_cov_matx[rep][time_i]
        joint_freq_change = np.add(joint_freq_change, df_frequency_change_dict['single']['amino_acid'][rep])
        joint_cov_mat = np.add(joint_cov_mat, invert_matrix)
        for k in range(q * L):
            invert_matrix[k,k] += gamma
        invert_matrix = np.linalg.inv(invert_matrix)
        estimate_selection[rep] = np.inner(invert_matrix, df_frequency_change_dict['single']['amino_acid'][rep])
    for k in range(q * L):
        joint_cov_mat[k,k] += gamma
    joint_selection = np.inner(np.linalg.inv(joint_cov_mat), joint_freq_change)
        
    df_column = ['site', 'amino_acid'] + ['rep_' + str(k) for k in replicates]
    df_selection_coefficients = pd.DataFrame(columns = df_column)
    column_site = []
    column_aa = []
    column_sc = []
    for rep in replicates:
        column_sc.append([])
    for site_name in sites:
        for aa in AA:
            column_site.append(site_name+1)
            column_aa.append(aa)
    for rep in replicates:
        column_sc[replicates.index(rep)].append(estimate_selection[rep])

    df_selection_coefficients['site'] = column_site
    df_selection_coefficients['amino_acid'] = column_aa
    for rep in replicates:
        df_selection_coefficients['rep_' + str(rep)] = column_sc[replicates.index(rep)][0]

    df_selection_coefficients['joint'] = joint_selection    
    
    return df_selection_coefficients

def optimize_regularization_full_genome(regularization_list, sites, cov_matx, df_frequency_change_dict, replicates):
    correlation_list = []
    
    for gamma in regularization_list:
        copy_cov_matx = copy.deepcopy(cov_matx['amino_acid'])
        q = len(AA)
        L = len(sites)
        estimate_selection = {}
        
        for rep in list(copy_cov_matx.keys()):
            
            invert_matrix = np.zeros((q * L, q * L))
            for k in range(q * L):
                invert_matrix[k,k] += gamma
            for i in range(len(list(copy_cov_matx[rep].keys()))-1):
                time_i = list(copy_cov_matx[rep].keys())[i]
                time_i_post = list(copy_cov_matx[rep].keys())[i+1]
                time_interval = time_i_post- time_i
                invert_matrix += time_interval * copy_cov_matx[rep][time_i]
            invert_matrix = np.linalg.inv(invert_matrix)
            estimate_selection[rep] = np.inner(invert_matrix, df_frequency_change_dict['single']['amino_acid'][rep])
        df_column = ['site', 'amino_acid'] + ['rep_' + str(k) for k in replicates]
        df_selection_coefficients = pd.DataFrame(columns = df_column)
        column_site = []
        column_aa = []
        column_sc = []
        for rep in replicates:
            column_sc.append([])
        for site_name in sites:
            for aa in AA:
                column_site.append(site_name+1)
                column_aa.append(aa)
        for rep in replicates:
            column_sc[replicates.index(rep)].append(estimate_selection[rep])

        df_selection_coefficients['site'] = column_site
        df_selection_coefficients['amino_acid'] = column_aa
        for rep in replicates:
            df_selection_coefficients['rep_' + str(rep)] = column_sc[replicates.index(rep)][0]

        correlation_list.append((df_selection_coefficients.iloc[:,2:].corr().sum().sum()-len(replicates))/(len(replicates)*(len(replicates)-1)))
    return correlation_list



def get_counts(Input_dir, replicates, Input_file_prefix, flag):

    Input_file_single = []
    df_allele_counts = {}
    
    if flag == 'single':
        for replicate in replicates:
            df_allele_counts[replicate] = {}
            Input_file = Input_dir + Input_file_prefix[flag] + str(replicate) + '.csv.zip'
            print(' ', Input_file)
            table = pd.read_csv(Input_file)
            original_rows = table.shape[0]
            indexNames = table[table['counts'] <= 0].index
            table.drop(indexNames , inplace = True)
            table.drop('replicate', axis = 1, inplace = True)
            current_rows = table.shape[0]
            
            table['codon_counts_position'] = table['codon'].apply(lambda a: CODONS.index(a))
            table['AA_counts_position'] = table['codon_counts_position'].apply(lambda a: codon_to_aa_num[a])
            table.drop('codon', axis = 1, inplace = True)

            for gen in table['generation'].unique().tolist():
                temp = table[table['generation'] == gen].copy()
                temp.drop('generation', axis = 1, inplace = True)
                df_allele_counts[replicate][gen] = temp           
            print('   delete %d rows of zero count records, remain %d rows of non zero count records' %(original_rows - current_rows, current_rows))

    if flag == 'double':
        for replicate in replicates:
            df_allele_counts[replicate] = {}
            Input_file = Input_dir + Input_file_prefix[flag] + str(replicate) + '.csv.zip'
            print(' ', Input_file)
            table = pd.read_csv(Input_file)
            original_rows = table.shape[0]
            indexNames = table[table['counts'] <= 0].index
            table.drop(indexNames , inplace = True)
            table.drop('replicate', axis = 1, inplace = True)
            current_rows = table.shape[0]
            table['codon_counts_position_1'] = table['codon_1'].apply(lambda a: CODONS.index(a))
            table.drop('codon_1', axis = 1, inplace = True)
            table['codon_counts_position_2'] = table['codon_2'].apply(lambda a: CODONS.index(a))
            table.drop('codon_2', axis = 1, inplace = True)
            table['AA_counts_position_1'] = table['codon_counts_position_1'].apply(lambda a: codon_to_aa_num[a])            
            table['AA_counts_position_2'] = table['codon_counts_position_2'].apply(lambda a: codon_to_aa_num[a])
            
            for gen in table['generation'].unique().tolist():
                temp = table[table['generation'] == gen].copy()
                temp.drop('generation', axis = 1, inplace = True)
                df_allele_counts[replicate][gen] = temp   
            print('   delete %d rows of zero count records, remain %d rows of non zero count records' %(original_rows - current_rows, current_rows))

    table = None
    temp = None
    indexNames = None
    return df_allele_counts

def counts_to_frequency(df_allele_counts_original, replicates, flag, sites):
    
    replicate_length = len(replicates)
    df_allele_frequency = {}
    
    max_read = 0
    
    if flag == 'single':
        print('Getting %s allele frequencies table:' %(flag))
        df_allele_frequency['amino_acid'] = {}
        replicates = df_allele_counts_original[flag].keys()
        for rep in replicates:
            df_allele_frequency['amino_acid'][rep] = {}
            print('Replicate %d' %rep)
            generations = list(df_allele_counts_original[flag][rep].keys())
            for gen in generations:
                single_aminoacid_counts_list = np.zeros(len(sites) * 21)
                single_aminoacid_frequency_list = np.zeros(len(sites) * 21)
                site_value = df_allele_counts_original[flag][rep][gen]['site'].tolist()
                site_index = [sites.index(k) for k in site_value]
                
                counts_value = df_allele_counts_original[flag][rep][gen]['counts'].tolist()
                AA_counts_index = df_allele_counts_original[flag][rep][gen]['AA_counts_position'].tolist()

                for i in range(len(AA_counts_index)):
                    single_aminoacid_counts_list[site_index[i]*21 + AA_counts_index[i]] += counts_value[i]                    
                
                for i in range(len(sites)):
                    total_reads = single_aminoacid_counts_list[i*21 : i*21+21].sum()  
                    single_aminoacid_frequency_list[i*21 : i*21+21] = single_aminoacid_counts_list[i*21 : i*21+21] / total_reads
                df_allele_frequency['amino_acid'][rep][gen] = single_aminoacid_frequency_list
                print('\tTotal %d reads in generation %d' %(total_reads, gen))
                max_read = max(total_reads, max_read)
                
    if flag == 'double':
        print('Getting %s allele frequencies table:' %(flag))
        df_allele_frequency['amino_acid'] = {}
        replicates = df_allele_counts_original[flag].keys()
        for rep in replicates:
            df_allele_frequency['amino_acid'][rep] = {}
            print('Replicate %d' %rep)
            generations = list(df_allele_counts_original[flag][rep].keys())
            for gen in generations:
                aminoacid_length = len(sites) * 21
                double_aminoacid_counts_list = np.zeros((aminoacid_length, aminoacid_length))
                double_aminoacid_frequency_list = np.zeros((aminoacid_length, aminoacid_length))
                
                counts_value = df_allele_counts_original[flag][rep][gen]['counts'].tolist()
                
                site_value_1 = df_allele_counts_original[flag][rep][gen]['site_1'].tolist()
                site_index_1 = [sites.index(k) for k in site_value_1]
                AA_counts_index_1 = df_allele_counts_original[flag][rep][gen]['AA_counts_position_1'].tolist()
                
                site_value_2 = df_allele_counts_original[flag][rep][gen]['site_2'].tolist()
                site_index_2 = [sites.index(k) for k in site_value_2]
                AA_counts_index_2 = df_allele_counts_original[flag][rep][gen]['AA_counts_position_2'].tolist()
                
                for i in range(len(AA_counts_index_1)):
                    double_aminoacid_counts_list[site_index_1[i]*21 + AA_counts_index_1[i], site_index_2[i]*21 + AA_counts_index_2[i]] += counts_value[i]
                    double_aminoacid_counts_list[site_index_2[i]*21 + AA_counts_index_2[i], site_index_1[i]*21 + AA_counts_index_1[i]] += counts_value[i]

                total_reads = double_aminoacid_counts_list[0:21, 21:42].sum()

                double_aminoacid_frequency_list = double_aminoacid_counts_list/total_reads
                
                df_allele_frequency['amino_acid'][rep][gen] = double_aminoacid_frequency_list
                
                print('\tTotal %d reads in generation %d' %(total_reads, gen))
                max_read = max(total_reads, max_read)
                
        aminoacid_length = None
        double_aminoacid_counts_list = None
        double_aminoacid_frequency_list = None                
        counts_value = None                
        site_value_1 = None
        AA_counts_index_1 = None                
        site_value_2 = None
        AA_counts_index_2 = None
    return (df_allele_frequency, max_read)

def frequency_change(df_allele_frequency, flag):
    
    df_allele_frequency_change = {}
    df_allele_frequency_change['amino_acid'] = {}
    additional_columns = ['initial_counts', 'initial_frequency', 'final_counts', 'final_frequency']
    replicates = list(df_allele_frequency[flag]['amino_acid'].keys())

    for rep in replicates:
        print('Getting %s allele frequency change of replicate %d between initial and final generation...' %(flag, rep))
        df_allele_frequency_change['amino_acid'][rep] = {}
        generations = list(df_allele_frequency[flag]['amino_acid'][rep].keys())
        initial_gen = min(generations)
        final_gen = max(generations)

        if flag == 'single':

            initial_site_frequency = df_allele_frequency[flag]['amino_acid'][rep][initial_gen].copy()
            final_site_frequency = df_allele_frequency[flag]['amino_acid'][rep][final_gen].copy()
            df_allele_frequency_change['amino_acid'][rep] = final_site_frequency - initial_site_frequency

        if flag == 'double':
            
            initial_site_frequency = df_allele_frequency[flag]['amino_acid'][rep][initial_gen].copy()
            final_site_frequency = df_allele_frequency[flag]['amino_acid'][rep][final_gen].copy()
            df_allele_frequency_change['amino_acid'][rep] = final_site_frequency - initial_site_frequency
            
    return df_allele_frequency_change
 
def covariance_matrix(df_frequency_dict, flag_list, sites):
    replicates = list(df_frequency_dict[flag_list[0]]['amino_acid'].keys())
    generations = list(df_frequency_dict[flag_list[0]]['amino_acid'][replicates[0]].keys())
    covariance_matrix_dict = {}
    covariance_matrix_dict['amino_acid'] = {}
    q = len(AA)
    L = len(sites)
    size_of_matrix = q * L
    for rep in replicates:
        covariance_matrix_dict['amino_acid'][rep] = {}
        for gen in range(len(generations[:-1])):   
            covariance_matrix_dict['amino_acid'][rep][generations[gen]] = (df_frequency_dict['double']['amino_acid'][rep][generations[gen]].copy()+df_frequency_dict['double']['amino_acid'][rep][generations[gen+1]].copy())/2.0
            # covariance_matrix_dict['amino_acid'][rep][gen] -= covariance_matrix_dict['amino_acid'][rep][gen]
            for i in range(size_of_matrix):
                site_i_freq_pre  = df_frequency_dict['single']['amino_acid'][rep][generations[gen]][i]
                site_i_freq_post = df_frequency_dict['single']['amino_acid'][rep][generations[gen+1]][i]
                first_term  = (3-2*site_i_freq_post)*(site_i_freq_post+site_i_freq_pre)/6
                second_term = site_i_freq_pre*site_i_freq_pre/3
                covariance_matrix_dict['amino_acid'][rep][generations[gen]][i, i] = first_term - second_term

                for j in range(i+1, size_of_matrix):
                    site_j_freq_pre  = df_frequency_dict['single']['amino_acid'][rep][generations[gen]][j]
                    site_j_freq_post = df_frequency_dict['single']['amino_acid'][rep][generations[gen+1]][j]
                    long_term = (2*site_i_freq_pre*site_j_freq_pre + 2*site_i_freq_post*site_j_freq_post + site_i_freq_pre*site_j_freq_post  + site_i_freq_post*site_j_freq_pre)/6
                    covariance_matrix_dict['amino_acid'][rep][generations[gen]][i, j] -= long_term
                    covariance_matrix_dict['amino_acid'][rep][generations[gen]][j, i] -= long_term
        covariance_matrix_dict['amino_acid'][rep][generations[-1]]={}
    return covariance_matrix_dict
    
def initialization(site_start, site_end, Input_dir, Output_dir, epistasis):
    Input_file_prefix = {}

    if epistasis is False:
        flag_list = ['single', 'double']

    else:
        flag_list = ['single', 'double', 'triple', 'quadruple']

    for flag in flag_list:
        Input_file_prefix[flag] = flag + '_allele_rep'

    sites = list(range(site_start, site_end+1))

    return Input_file_prefix, flag_list, sites

