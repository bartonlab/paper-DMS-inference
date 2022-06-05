import sys, os
#from copy import deepcopy
from importlib import reload

import numpy as np
import itertools

import scipy as sp
import scipy.stats as st
from scipy import interpolate 
import copy

import pandas as pd

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

import seaborn as sns

import tqdm
import time
import csv
import math



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



# def get_counts_old(Input_dir, replicates, Input_file_prefix, epistasis):

#     Input_file_single = []
#     Input_file_double = []
#     df_single_allele_counts = []
#     df_double_allele_counts = []
#     if epistasis is False:
#         for replicate in replicates:
#             Input_file_single = Input_dir + Input_file_prefix['single'] + str(replicate) + '.csv'
#             Input_file_double = Input_dir + Input_file_prefix['double'] + str(replicate) + '.csv'
#             single_table = pd.read_csv(Input_file_single)
#             double_table = pd.read_csv(Input_file_double)
#             df_single_allele_counts.append(single_table)
#             df_double_allele_counts.append(double_table)
#             print('Reading counts files from:\n', Input_file_single, '\n', Input_file_double)
    
#     return (df_single_allele_counts, df_double_allele_counts)
    



def get_counts(Input_dir, replicates, Input_file_prefix, flag):

    Input_file_single = []
    df_allele_counts = {}
    
    if flag == 'single':
        for replicate in replicates:
            df_allele_counts[replicate] = {}
            Input_file = Input_dir + Input_file_prefix[flag] + str(replicate) + '.csv'
            print(' ', Input_file)
            table = pd.read_csv(Input_file)
            original_rows = table.shape[0]
            indexNames = table[table['counts'] <= 0].index
            table.drop(indexNames , inplace = True)
            table.drop('replicate', axis = 1, inplace = True)
            current_rows = table.shape[0]
            #table['amino_acid'] = table['codon'].apply(codon2aa)
            
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
            Input_file = Input_dir + Input_file_prefix[flag] + str(replicate) + '.csv'
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
    
    #print(sites)
    replicate_length = len(replicates)
    df_allele_frequency = {}
    
    max_read = 0
    
    if flag == 'single':
        print('Getting %s allele frequencies table:' %(flag))
#        df_allele_frequency['codon'] = {}
        df_allele_frequency['amino_acid'] = {}
        replicates = df_allele_counts_original[flag].keys()
        for rep in replicates:
#            df_allele_frequency['codon'][rep] = {}
            df_allele_frequency['amino_acid'][rep] = {}
            print('Replicate %d' %rep)
            generations = list(df_allele_counts_original[flag][rep].keys())
            for gen in generations:

#                single_codon_counts_list = np.zeros(len(sites) * 64)
                single_aminoacid_counts_list = np.zeros(len(sites) * 21)
#                single_codon_frequency_list = np.zeros(len(sites) * 64)
                single_aminoacid_frequency_list = np.zeros(len(sites) * 21)
                site_value = df_allele_counts_original[flag][rep][gen]['site'].tolist()
                site_index = [sites.index(k) for k in site_value]
                
                counts_value = df_allele_counts_original[flag][rep][gen]['counts'].tolist()
                AA_counts_index = df_allele_counts_original[flag][rep][gen]['AA_counts_position'].tolist()
#                codon_counts_index = df_allele_counts_original[flag][rep][gen]['codon_counts_position'].tolist()

                for i in range(len(AA_counts_index)):
                    single_aminoacid_counts_list[site_index[i]*21 + AA_counts_index[i]] += counts_value[i]
#                for i in range(len(codon_counts_index)):
#                    single_codon_counts_list[(site_value[i]-1)*64 +codon_counts_index[i]] += counts_value[i]   
                    
                
                for i in range(len(sites)):
                    total_reads = single_aminoacid_counts_list[i*21 : i*21+21].sum()  
#                    single_codon_frequency_list[i*64 : i*64+64] = single_codon_counts_list[i*64 : i*64+64] / total_reads
                    single_aminoacid_frequency_list[i*21 : i*21+21] = single_aminoacid_counts_list[i*21 : i*21+21] / total_reads
                                    
                    
#                df_allele_frequency['codon'][rep][gen] = single_codon_frequency_list
                df_allele_frequency['amino_acid'][rep][gen] = single_aminoacid_frequency_list
                #return None
                

                print('\tTotal %d reads in generation %d' %(total_reads, gen))
                max_read = max(total_reads, max_read)
                #return single_codon_frequency_list, single_codon_counts_list

                
                
    if flag == 'double':
        print('Getting %s allele frequencies table:' %(flag))
#        df_allele_frequency['codon'] = {}
        df_allele_frequency['amino_acid'] = {}
        replicates = df_allele_counts_original[flag].keys()
        for rep in replicates:
#            df_allele_frequency['codon'][rep] = {}
            df_allele_frequency['amino_acid'][rep] = {}
            print('Replicate %d' %rep)
            generations = list(df_allele_counts_original[flag][rep].keys())
            for gen in generations:
#                codon_length = len(sites) * 64
#                double_codon_counts_list = np.zeros((codon_length, codon_length))
#                double_codon_frequency_list = np.zeros((codon_length, codon_length))
#                codon_counts_index_1 = df_allele_counts_original[flag][rep][gen]['codon_counts_position_1'].tolist()
#                codon_counts_index_2 = df_allele_counts_original[flag][rep][gen]['codon_counts_position_2'].tolist()
#                for i in range(len(codon_counts_index_1)):
#                    double_codon_counts_list[(site_value_1[i]-1)*64 + codon_counts_index_1[i], (site_value_2[i]-1)*64 + codon_counts_index_2[i]] += counts_value[i]
#                    double_codon_counts_list[(site_value_2[i]-1)*64 + codon_counts_index_2[i], (site_value_1[i]-1)*64 + codon_counts_index_1[i]] += counts_value[i]
#                double_codon_frequency_list = double_codon_counts_list/total_reads
#                df_allele_frequency['codon'][rep][gen] = double_codon_frequency_list
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
#     max_read = 0
#     print(max_read)
    return (df_allele_frequency, max_read)




# def frequency_change_old(df_allele_frequency, flag):
    
#     df_allele_frequency_change = {}
#     replicates = list(df_allele_frequency.keys())

#     for rep in replicates:
#         df_allele_frequency_change[rep] = {}
#         generations = list(df_allele_frequency[rep].keys())
#         initial_gen = min(generations)
#         end_gen = max(generations)
        
#         if flag == 'single':
#             print('Getting %s allele frequency change of replicate %d between initial and final generation...' %(flag, rep))
#             #sites = list(df_allele_frequency[rep][initial_gen].keys())
#             for site in sites:
#                 initial_site_frequency = df_allele_frequency[rep][initial_gen][site].copy()
#                 end_site_frequency = df_allele_frequency[rep][end_gen][site].copy()
#                 df_allele_frequency_change[rep][site] = df_allele_frequency[rep][initial_gen][site].copy()
#                 df_allele_frequency_change[rep][site] = df_allele_frequency_change[rep][site].drop(['counts', 'frequency'], axis = 1)
#                 df_allele_frequency_change[rep][site]['initial_counts'] = initial_site_frequency['counts'].values
#                 df_allele_frequency_change[rep][site]['initial_frequency'] = initial_site_frequency['frequency'].values
#                 df_allele_frequency_change[rep][site]['end_counts'] = end_site_frequency['counts'].values
#                 df_allele_frequency_change[rep][site]['end_frequency'] = end_site_frequency['frequency'].values
#                 df_allele_frequency_change[rep][site]['frequency_change'] = df_allele_frequency_change[rep][site]['end_frequency'] - df_allele_frequency_change[rep][site]['initial_frequency']  

#         elif flag == 'double':
#             print('Getting %s allele frequency change of replicate %d between initial and final generation...' %(flag, rep))
#             #sites = list(df_allele_frequency[rep][initial_gen].keys())
#             for site in sites:
#                 df_allele_frequency_change[rep][site] = {}
#             for site_1, site_2 in itertools.combinations(sites, 2):
#                 #print(site_1, site_2)
#                 initial_site_frequency = df_allele_frequency[rep][initial_gen][site_1][site_2].copy()
#                 end_site_frequency = df_allele_frequency[rep][end_gen][site_1][site_2].copy()
#                 df_allele_frequency_change[rep][site_1][site_2] = df_allele_frequency[rep][initial_gen][site_1][site_2].copy()
#                 df_allele_frequency_change[rep][site_1][site_2] = df_allele_frequency_change[rep][site_1][site_2].drop(['counts', 'frequency'], axis = 1)
#                 df_allele_frequency_change[rep][site_1][site_2]['initial_counts'] = initial_site_frequency['counts'].values
#                 df_allele_frequency_change[rep][site_1][site_2]['initial_frequency'] = initial_site_frequency['frequency'].values
#                 df_allele_frequency_change[rep][site_1][site_2]['end_counts'] = end_site_frequency['counts'].values
#                 df_allele_frequency_change[rep][site_1][site_2]['end_frequency'] = end_site_frequency['frequency'].values
#                 df_allele_frequency_change[rep][site_1][site_2]['frequency_change'] = df_allele_frequency_change[rep][site_1][site_2]['end_frequency'] - df_allele_frequency_change[rep][site_1][site_2]['initial_frequency']
#         else:
#             print('Incorrect allele frequencies request, please input "flag" as one of ["single", "double", "triple", "quadruple"]')
#     return df_allele_frequency_change    


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
    # print(generations)
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
    # print([covariance_matrix_dict['amino_acid'][1][generations[gen]][k,k] for k in range(q*L)])
    return covariance_matrix_dict
    

def initialization(site_start, site_end, Input_dir, Output_dir, epistasis):
    Input_file_prefix = {}

    if epistasis is False:
        flag_list = ['single', 'double']


    else:
        flag_list = ['single', 'double', 'triple', 'quadruple']

    for flag in flag_list:
        Input_file_prefix[flag] = flag + '_allele_counts_rep'

    sites = list(range(site_start, site_end+1))

    return Input_file_prefix, flag_list, sites


NUC = ['-', 'A', 'C', 'G', 'T']                           # Nucleotide letter
REF = NUC[0]

CODONS = ['AAA', 'AAC', 'AAG', 'AAT', 'ACA', 'ACC', 'ACG', 'ACT',   # Tri-nucleotide units table
          'AGA', 'AGC', 'AGG', 'AGT', 'ATA', 'ATC', 'ATG', 'ATT',
          'CAA', 'CAC', 'CAG', 'CAT', 'CCA', 'CCC', 'CCG', 'CCT',
          'CGA', 'CGC', 'CGG', 'CGT', 'CTA', 'CTC', 'CTG', 'CTT',
          'GAA', 'GAC', 'GAG', 'GAT', 'GCA', 'GCC', 'GCG', 'GCT',
          'GGA', 'GGC', 'GGG', 'GGT', 'GTA', 'GTC', 'GTG', 'GTT',
          'TAA', 'TAC', 'TAG', 'TAT', 'TCA', 'TCC', 'TCG', 'TCT',
          'TGA', 'TGC', 'TGG', 'TGT', 'TTA', 'TTC', 'TTG', 'TTT']   

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


AA  = sorted(['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I','L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '*'])
NUC = ['A', 'C', 'G', 'T']

codon_to_aa_num = {}
for i in range(len(CODONS)):
    codon = CODONS[i]
    aminoacid = codon2aa(codon)
    aminoacid_num = AA.index(aminoacid)
    codon_to_aa_num[i] = aminoacid_num

