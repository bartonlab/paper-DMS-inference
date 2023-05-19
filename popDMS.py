import sys, os
import re
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


# Get frequency intermediate data file from raw variant table

def nucleotide_file_to_counts_single_allele(nucleotide_file_name, reference_sequence, timepoints, rep, table_col_name, save_path):
    codon_length = 3
    df_data = pd.read_csv(nucleotide_file_name, compression='gzip')
    df_data = df_data.fillna(0)
    df_data.reset_index(drop=True, inplace=True)
    df_data = df_data.drop(df_data.columns[0], axis = 1)
    
    if 'TpoR' in nucleotide_file_name:
        df_data = df_data.drop(['hgvs_splice'], axis=1)
      
    df_data[df_data['hgvs_nt'].str.contains('X', regex=False)]
    df_data = df_data[~df_data.hgvs_nt.str.contains('X', regex=False)]
    
    df_frequency = df_data.loc[:,df_data.columns[2]:].astype('float')
    df_frequency.loc[:,df_frequency.columns[2]:] = df_frequency.loc[:,df_frequency.columns[2]:].div(df_frequency.sum(axis=1),axis=0)
    site_list = sorted(df_data['hgvs_pro'].str.extractall('(\d+)')[0].astype(int).unique())
    
    raw_codon = [reference_sequence[i:i+codon_length] for i in range(0, len(reference_sequence), codon_length)]
    allele_counts_columns = ['generation', 'site', 'codon', 'counts']
    allele_counts_table = df_data[table_col_name]

    temp = allele_counts_table.columns.tolist()[0]
    allele_counts_table = allele_counts_table.rename(columns={temp: 'variants'})
    count_table_columns = allele_counts_table.columns.tolist()
    allele_counts_table[allele_counts_table['variants'] == '_wt']

    allele_counts_table_no_wt = allele_counts_table.drop(allele_counts_table.index[allele_counts_table['variants'] == '_wt'])

    total_count = []
    total_mut = []
    total_wt  = []

    for i in range(len(timepoints)):
        counts_all = allele_counts_table[count_table_columns[i+1]].tolist()
        temp = [int(integer) for integer in counts_all]
        counts_all = temp
        summation_all = sum(counts_all)
        total_count.append(summation_all)
        counts_mut = allele_counts_table_no_wt[count_table_columns[i+1]].tolist()
        temp = [int(integer) for integer in counts_mut]
        counts_mut = temp
        summation_mut = sum(counts_mut)
        total_mut.append(summation_mut)
        summation_wt = summation_all - summation_mut
        total_wt.append(summation_wt)
    
    codon_allele_dict = {}
    for gen in timepoints:
        codon_allele_dict[gen] = {}
        for idx in site_list:
            codon_allele_dict[gen][idx] = {}
            for codon in CODONS:
                codon_allele_dict[gen][idx][codon] = 0

            codon_allele_dict[gen][idx][raw_codon[site_list.index(idx)]] = total_count[timepoints.index(gen)]
    
    reference_list = list(reference_sequence)
    for i in range(allele_counts_table_no_wt.shape[0]):
        print("Progress {:2.1%}".format(i / allele_counts_table_no_wt.shape[0]), end="\r")
        variants_allele = allele_counts_table_no_wt.iloc[i].variants
        mutation_number = allele_counts_table_no_wt.iloc[i].tolist()[1:]
        temp = [int(integer) for integer in mutation_number]
        mutation_number = temp
        nucleotide = [x for x in variants_allele if x.isalpha()]
        variant_site = re.findall("(\d+)", variants_allele)
        nucleotide = nucleotide[1:]
        variant_list = reference_list.copy()
        for j in range(len(variant_site)):
            variant_list[int(variant_site[j])-1] = nucleotide[2 * j + 1]
        variant_sequence = ''.join(variant_list)
        variant_codon = [variant_sequence[i:i+codon_length] for i in range(0, len(variant_sequence), codon_length)]
        for r, n, idx in zip(raw_codon, variant_codon, site_list):
            if r != n:
                for gen in timepoints:
                    codon_allele_dict[gen][idx][r] -= mutation_number[timepoints.index(gen)]
                    codon_allele_dict[gen][idx][n] += mutation_number[timepoints.index(gen)]

    allele_counts_list = []
    for gen, site_codon_counts in codon_allele_dict.items():
        for site, codon_counts in site_codon_counts.items():
            for codon, counts in codon_counts.items():
                allele_counts_list.append([gen, site, codon, counts])

    codon_counts_table = pd.DataFrame(data = allele_counts_list, columns = allele_counts_columns)
    codon_counts_table.to_csv(save_path, sep = ',', index = False, compression = 'gzip')
    return site_list


def nucleotide_file_to_counts_double_allele(nucleotide_file_name, reference_sequence, timepoints, rep, table_col_name, save_path):
    codon_length = 3
    df_data = pd.read_csv(nucleotide_file_name, compression='gzip')
    df_data = df_data.fillna(0)
    df_data.reset_index(drop=True, inplace=True)
    df_data = df_data.drop(df_data.columns[0], axis = 1)
    
    if 'TpoR' in nucleotide_file_name:
        df_data = df_data.drop(['hgvs_splice'], axis=1)

    df_data[df_data['hgvs_nt'].str.contains('X', regex=False)]
    df_data = df_data[~df_data.hgvs_nt.str.contains('X', regex=False)]
    
    df_frequency = df_data.loc[:,df_data.columns[2]:].astype('float')
    df_frequency.loc[:,df_frequency.columns[2]:] = df_frequency.loc[:,df_frequency.columns[2]:].div(df_frequency.sum(axis=1),axis=0)
    site_list = sorted(df_data['hgvs_pro'].str.extractall('(\d+)')[0].astype(int).unique())
    raw_codon = [reference_sequence[i:i+codon_length] for i in range(0, len(reference_sequence), codon_length)]
    allele_counts_columns = ['generation', 'site_1', 'codon_1', 'site_2', 'codon_2', 'counts']
    allele_counts_table = df_data[table_col_name]

    temp = allele_counts_table.columns.tolist()[0]
    allele_counts_table = allele_counts_table.rename(columns={temp: 'variants'})
    count_table_columns = allele_counts_table.columns.tolist()
    allele_counts_table[allele_counts_table['variants'] == '_wt']

    allele_counts_table_no_wt = allele_counts_table.drop(allele_counts_table.index[allele_counts_table['variants'] == '_wt'])

    total_count = []
    total_mut = []
    total_wt  = []
    for i in range(len(timepoints)):
        counts_all = allele_counts_table[count_table_columns[i+1]].tolist()
        temp = [int(integer) for integer in counts_all]
        counts_all = temp
        summation_all = sum(counts_all)
        total_count.append(summation_all)
        counts_mut = allele_counts_table_no_wt[count_table_columns[i+1]].tolist()
        temp = [int(integer) for integer in counts_mut]
        counts_mut = temp
        summation_mut = sum(counts_mut)
        total_mut.append(summation_mut)
        summation_wt = summation_all - summation_mut
        total_wt.append(summation_wt)

    length_site_list = len(site_list)
    length_codon_list = len(CODONS)
    codon_allele_dict = {}
    for gen in timepoints:
        codon_allele_dict[gen] = {}
        for idx_i in range(length_site_list):
            codon_allele_dict[gen][site_list[idx_i]] = {}
            codon_allele_dict[gen][site_list[idx_i]][raw_codon[idx_i]] = {}
            for idx_j in range(idx_i+1, length_site_list):
                codon_allele_dict[gen][site_list[idx_i]][raw_codon[idx_i]][site_list[idx_j]] = {}
                codon_allele_dict[gen][site_list[idx_i]][raw_codon[idx_i]][site_list[idx_j]][raw_codon[idx_j]] = total_count[timepoints.index(gen)]

    reference_list = list(reference_sequence)

    for i in range(allele_counts_table_no_wt.shape[0]):
        print("Progress {:2.1%}".format(i / allele_counts_table_no_wt.shape[0]), end="\r")
        variants_allele = allele_counts_table_no_wt.iloc[i].variants
        mutation_number = allele_counts_table_no_wt.iloc[i].tolist()[1:]
        temp = [int(integer) for integer in mutation_number]
        mutation_number = temp
        nucleotide = [x for x in variants_allele if x.isalpha()]
        variant_site = re.findall("(\d+)", variants_allele)
        nucleotide = nucleotide[1:]
        variant_list = reference_list.copy()
        for j in range(len(variant_site)):
            variant_list[int(variant_site[j])-1] = nucleotide[2 * j + 1]
        variant_sequence = ''.join(variant_list)
        variant_codon = [variant_sequence[i:i+codon_length] for i in range(0, len(variant_sequence), codon_length)]
        variant_site = []

        for r, n, idx in zip(raw_codon, variant_codon, site_list):
            if r != n:
                variant_site.append(idx)

        double_variant = []
        for v_site in variant_site:
            v_site_index = site_list.index(v_site)
            for j in range(v_site_index):
                double_variant.append([site_list[j], v_site])
            for j in range(v_site_index+1, len(site_list)):
                double_variant.append([v_site, site_list[j]])

        for d_v in double_variant:
            site_i = d_v[0]
            site_j = d_v[1]
            idx_i = site_list.index(site_i)
            idx_j = site_list.index(site_j)
            codon_i = variant_codon[idx_i]
            codon_j = variant_codon[idx_j]

            for gen in timepoints:
                codon_allele_dict[gen][site_i][raw_codon[idx_i]][site_j][raw_codon[idx_j]] -= mutation_number[timepoints.index(gen)]
                if codon_i not in codon_allele_dict[gen][site_i].keys():
                    codon_allele_dict[gen][site_i][codon_i] = {}
                    codon_allele_dict[gen][site_i][codon_i][site_j] = {}
                    codon_allele_dict[gen][site_i][codon_i][site_j][codon_j] = mutation_number[timepoints.index(gen)]
                else:
                    if site_j not in codon_allele_dict[gen][site_i][codon_i].keys():
                        codon_allele_dict[gen][site_i][codon_i][site_j] = {}
                        codon_allele_dict[gen][site_i][codon_i][site_j][codon_j] = mutation_number[timepoints.index(gen)]
                    else:
                        if codon_j not in codon_allele_dict[gen][site_i][codon_i][site_j].keys():
                            codon_allele_dict[gen][site_i][codon_i][site_j][codon_j] = mutation_number[timepoints.index(gen)]
                        else:
                            codon_allele_dict[gen][site_i][codon_i][site_j][codon_j] += mutation_number[timepoints.index(gen)]

        if len(variant_site)>1:
            error_deletion = list(combinations(variant_site, 2))
            for item in error_deletion:
                site_i = item[0]
                site_j = item[1]
                idx_i = site_list.index(site_i)
                idx_j = site_list.index(site_j)
                codon_i = variant_codon[idx_i]
                codon_j = variant_codon[idx_j]
                for gen in timepoints:
                    codon_allele_dict[gen][site_i][raw_codon[idx_i]][site_j][raw_codon[idx_j]] += mutation_number[timepoints.index(gen)]
                    codon_allele_dict[gen][site_i][codon_i][site_j][codon_j] -= mutation_number[timepoints.index(gen)]


    allele_counts_list = []
    for gen, site_codon_counts in codon_allele_dict.items():
        for site_i, codoni_sitej_codonj_countj in site_codon_counts.items():
            for codon_i, sitej_codonj_countj in codoni_sitej_codonj_countj.items():
                for site_j, codonj_countj in sitej_codonj_countj.items():
                    for codon_j, count_j in codonj_countj.items():
                        allele_counts_list.append([gen, site_i, codon_i, site_j, codon_j, count_j])


    codon_counts_table = pd.DataFrame(data = allele_counts_list, columns = allele_counts_columns)
    codon_counts_table.to_csv(save_path, sep = ',', index = False, compression = 'gzip')
    return True

# TARGET_PROTEIN_NAME, REFERENCE_SEQUENCE, RAW_HAPLOTYPE, TRANSFORMED_HAPLOTYPE, REPLICATES, INPUT_COUNTS_DIR, GENERATIONS, HAPLOTYPE_REP_GEN_COL
def output_save_allele(TARGET_PROTEIN_NAME, REFERENCE_SEQUENCE, RAW_HAPLOTYPE, TRANSFORMED_HAPLOTYPE, REPLICATES, GENERATIONS, INPUT_COUNTS_DIR, HAPLOTYPE_REP_GEN_COL):
    with open(REFERENCE_SEQUENCE,"r") as raw_seq:
        REFER_SEQ = raw_seq.read()
    REP_LIST = [i+1 for i in range(REPLICATES)]
    df_count = pd.read_csv(RAW_HAPLOTYPE, skiprows=4)
    df_count.to_csv(TRANSFORMED_HAPLOTYPE, compression='gzip', index = False)
    for REP in REP_LIST:
        print(TARGET_PROTEIN_NAME + ', replicate '+str(REP)+', single allele collecting')
        SAVE_PATH = INPUT_COUNTS_DIR + TARGET_PROTEIN_NAME + '_single_allele_rep' + str(REP) + '.csv.gzip' 
        site_list = nucleotide_file_to_counts_single_allele(TRANSFORMED_HAPLOTYPE, REFER_SEQ, GENERATIONS, REP, HAPLOTYPE_REP_GEN_COL[REP], SAVE_PATH)
        print(TARGET_PROTEIN_NAME + ', replicate '+str(REP)+', single allele finished')
        print(TARGET_PROTEIN_NAME + ', replicate '+str(REP)+', double allele collecting')
        SAVE_PATH = INPUT_COUNTS_DIR + TARGET_PROTEIN_NAME + '_double_allele_rep' + str(REP) + '.csv.gzip'
        nucleotide_file_to_counts_double_allele(TRANSFORMED_HAPLOTYPE, REFER_SEQ, GENERATIONS, REP, HAPLOTYPE_REP_GEN_COL[REP], SAVE_PATH)
        print(TARGET_PROTEIN_NAME + ', replicate '+str(REP)+', double allele finished')

    return site_list

# codes for independent site data

def independent_site_pipeline(TARGET_PROTEIN, REPLICATES, OUTPUT_DIR, DNACODON_FILE, PRE_FILES, POST_FILES, EPISTASIS, REGULARIZATION_PERCENT):
    estimate_selection, regularization_list = MPL_independent_site_inference(TARGET_PROTEIN, REPLICATES, DNACODON_FILE, PRE_FILES, POST_FILES)
    correlation_list = optimize_regularization_independent_site(estimate_selection, REPLICATES, regularization_list)
    PLOT_REGULARIZATION_CORRELATION(regularization_list, correlation_list, TARGET_PROTEIN)
    REGULARIZATION_SELECTED = find_best_regularization(regularization_list, correlation_list, REGULARIZATION_PERCENT)
    FINAL_SELECTION = output_final_selection_independent_site(REGULARIZATION_SELECTED, DNACODON_FILE, estimate_selection, REPLICATES)
    if os.path.exists(OUTPUT_DIR):
        FINAL_SELECTION.to_csv(OUTPUT_DIR+TARGET_PROTEIN+'.csv.gz', index = False, compression = 'gzip')
    else:
        os.makedirs(OUTPUT_DIR)
        FINAL_SELECTION.to_csv(OUTPUT_DIR+TARGET_PROTEIN+'.csv.gz', index = False, compression = 'gzip')
    with open(OUTPUT_DIR+TARGET_PROTEIN + '_supplementary.log', 'w') as f:
        f.write('Optimized regularization = %.6f' %REGULARIZATION_SELECTED)
    # reg =  %int(np.log10(REGULARIZATION_SELECTED))


def optimize_regularization_independent_site(estimate_selection, REPLICATES, REGULARIZATION_LIST):
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

def PLOT_REGULARIZATION_CORRELATION(regularization_list, correlation_list, protein_name):
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

def output_final_selection_independent_site(REGULARIZATION_SELECTED, DNACODON_FILE, estimate_selection, REPLICATES):
    
    df_0 = pd.read_csv('%s' %DNACODON_FILE, comment = '#', memory_map = True)
    site_list = []
    for site in df_0['site']:
        site_list+=[site]*21  
    AA_list = AA*len(df_0['site'])
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
        
def MPL_independent_site_inference(TARGET_PROTEIN, REPLICATES, DNACODON_FILE, PRE_FILE, POST_FILE):
    print('------ Calculating single replicate selection coefficients for %s ------' %TARGET_PROTEIN)
    print("\nCalculating error probability from %s...\n" %DNACODON_FILE)

    err = err_correct(DNACODON_FILE)
    estimate_selection = {}
    
    minimum = sys.float_info.max
    
    file_idx = 0
    for run_name in REPLICATES:
        run_name = str(run_name)
        estimate_selection[run_name]={}
        func_para = {
            'PRE_FILE':     PRE_FILE[file_idx],           
            'POST_FILE':    POST_FILE[file_idx],        
            'run_name':     run_name,                                            # Replicate serial number
            'homolog':      TARGET_PROTEIN,                                      # Homolog name
            'err':          err                                                  # Error probability for this homolog
        }
        file_idx += 1
        FREQUENCY_DIFF, COVARIANCE_MATRIX, L, q, max_read = MPL_independent_site_elements(**func_para)
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



def err_correct(DNACODON_FILE):

    df_0 = pd.read_csv('%s' %DNACODON_FILE, comment = '#', memory_map = True)     # Read raw data file

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

def MPL_independent_site_elements(**func_para):
 
    PRE_FILE     = func_para['PRE_FILE']
    POST_FILE    = func_para['POST_FILE']
    run_name     = func_para['run_name']
    homolog      = func_para['homolog']
    err          = func_para['err']

    print("Calculating selection coefficients for replicate_%s of %s:" %(run_name,homolog))

    df_0 = pd.read_csv('%s' % PRE_FILE,  comment = '#', memory_map = True)
    df_1 = pd.read_csv('%s' % POST_FILE, comment = '#', memory_map = True)

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



# codes for full length data

# TARGET_PROTEIN_NAME, SITE_LIST, REPLICATES, INPUT_COUNTS_DIR, OUTPUT_SELECTION_DIR, REGULARIZATION_PERCENT
# TARGET_PROTEIN_NAME, SITE_LIST, REPLICATES, INPUT_COUNTS_DIR, OUTPUT_SELECTION_DIR, REGULARIZATION_PERCENT
def full_length_pipeline(TARGET_PROTEIN_NAME, REPLICATE, INPUT_COUNTS_DIR, OUTPUT_SELECTION_DIR, REGULARIZATION_PERCENT):
    EPISTASIS = False
    INPUT_FILE_PREFIX, FLAG_LIST, REPLICATES = initialization(REPLICATE, EPISTASIS)
    FREQUENCY_DIFF, COVARIANCE_MATRIX, MAX_READ, SITE_LIST = MPL_full_length_elements(TARGET_PROTEIN_NAME, FLAG_LIST, INPUT_COUNTS_DIR, REPLICATES, INPUT_FILE_PREFIX)
    REGULARIZATION_LIST = [np.power(10, float(i)) for i in range(int(np.log10(1/MAX_READ)-1), 4)]
    CORRELATION_LIST = optimize_regularization_full_length(REGULARIZATION_LIST, SITE_LIST, COVARIANCE_MATRIX, FREQUENCY_DIFF, REPLICATES)
    PLOT_REGULARIZATION_CORRELATION(REGULARIZATION_LIST, CORRELATION_LIST, TARGET_PROTEIN_NAME)
    REGULARIZATION_SELECTED = find_best_regularization(REGULARIZATION_LIST, CORRELATION_LIST, REGULARIZATION_PERCENT)
    FINAL_SELECTION = output_final_selection_full_length(REGULARIZATION_SELECTED, SITE_LIST, COVARIANCE_MATRIX, FREQUENCY_DIFF, REPLICATES)
    if os.path.exists(OUTPUT_SELECTION_DIR):
        FINAL_SELECTION.to_csv(OUTPUT_SELECTION_DIR + TARGET_PROTEIN_NAME + '.csv.gz', index = False, compression = 'gzip')
    else:
        os.makedirs(OUTPUT_SELECTION_DIR)
        FINAL_SELECTION.to_csv(OUTPUT_SELECTION_DIR + TARGET_PROTEIN_NAME + '.csv.gz', index = False, compression = 'gzip')

    with open(OUTPUT_SELECTION_DIR+TARGET_PROTEIN_NAME + '_supplementary.log', 'w') as f:
        f.write('Optimized regularization = %.e' %REGULARIZATION_SELECTED)

def MPL_full_length_elements(Target_protein, flag_list, Input_dir, replicates, Input_file_prefix):
    df_counts_dict = {}
    df_frequency_dict = {}
    df_frequency_change_dict = {}
    max_read_final = 0
    for flag in flag_list:
        print('Reading %s allele counts files from:' %flag)
        df_counts_dict[flag], site_list = get_counts(Target_protein, Input_dir, replicates, Input_file_prefix, flag)
    for flag in flag_list:
        df_frequency_dict[flag], max_read = counts_to_frequency(df_counts_dict, replicates, flag, site_list)
        max_read_final = max(max_read_final, max_read)
    for flag in flag_list:
        df_frequency_change_dict[flag] = frequency_change(df_frequency_dict, flag)
    cov_matx = covariance_matrix(df_frequency_dict, flag_list, site_list)   
    return df_frequency_change_dict, cov_matx, max_read_final, site_list

def output_final_selection_full_length(gamma, sites, cov_matx, df_frequency_change_dict, replicates):
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
            column_site.append(site_name)
            column_aa.append(aa)
    for rep in replicates:
        column_sc[replicates.index(rep)].append(estimate_selection[rep])
    df_selection_coefficients['site'] = column_site
    df_selection_coefficients['amino_acid'] = column_aa
    for rep in replicates:
        df_selection_coefficients['rep_' + str(rep)] = ['{:.2E}'.format(i) for i in column_sc[replicates.index(rep)][0]]
    df_selection_coefficients['joint'] = ['{:.2E}'.format(i) for i in joint_selection]   
    return df_selection_coefficients

def optimize_regularization_full_length(regularization_list, sites, cov_matx, df_frequency_change_dict, replicates):
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



def get_counts(Target_protein, Input_dir, replicates, Input_file_prefix, flag):

    Input_file_single = []
    df_allele_counts = {}
    
    if flag == 'single':
        for replicate in replicates:
            df_allele_counts[replicate] = {}
            Input_file = Input_dir+Target_protein+'_' + Input_file_prefix[flag] + str(replicate) + '.csv.gzip'
            print(' ', Input_file)
            table = pd.read_csv(Input_file, compression='gzip')
            site_list = sorted(table['site'].unique())
            original_rows = table.shape[0]
            indexNames = table[table['counts'] <= 0].index
            table.drop(indexNames , inplace = True)
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
            Input_file = Input_dir+Target_protein+'_' + Input_file_prefix[flag] + str(replicate) + '.csv.gzip'
            print(' ', Input_file)
            table = pd.read_csv(Input_file, compression='gzip')
            site_list = sorted(np.unique(table[['site_1', 'site_2']].values))
            original_rows = table.shape[0]
            indexNames = table[table['counts'] <= 0].index
            table.drop(indexNames , inplace = True)
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
    return df_allele_counts, site_list

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
    
def initialization(replicate, epistasis):
    Input_file_prefix = {}
    replicates = [i+1 for i in range(replicate)]

    if epistasis is False:
        flag_list = ['single', 'double']

    else:
        flag_list = ['single', 'double', 'triple', 'quadruple']

    for flag in flag_list:
        Input_file_prefix[flag] = flag + '_allele_rep'

    return Input_file_prefix, flag_list, replicates

