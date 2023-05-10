import numpy as np
import pandas as pd
import csv
import re
import time
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations

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


df_data = pd.read_csv('../data/raw_data/YAP1_nucleotide_variant.csv', low_memory=False)
df_data = df_data.fillna(0)
df_data = df_data.drop([0, 1, 2], axis = 0)
df_data = df_data.drop(df_data.columns[0], axis = 1)
df_data.reset_index(drop=True, inplace=True)
df_data.columns = df_data.iloc[0]
df_data = df_data.drop([0], axis = 0)

CODONS = ['AAA', 'AAC', 'AAG', 'AAT', 'ACA', 'ACC', 'ACG', 'ACT',   # Tri-nucleotide units table
          'AGA', 'AGC', 'AGG', 'AGT', 'ATA', 'ATC', 'ATG', 'ATT',
          'CAA', 'CAC', 'CAG', 'CAT', 'CCA', 'CCC', 'CCG', 'CCT',
          'CGA', 'CGC', 'CGG', 'CGT', 'CTA', 'CTC', 'CTG', 'CTT',
          'GAA', 'GAC', 'GAG', 'GAT', 'GCA', 'GCC', 'GCG', 'GCT',
          'GGA', 'GGC', 'GGG', 'GGT', 'GTA', 'GTC', 'GTG', 'GTT',
          'TAA', 'TAC', 'TAG', 'TAT', 'TCA', 'TCC', 'TCG', 'TCT',
          'TGA', 'TGC', 'TGG', 'TGT', 'TTA', 'TTC', 'TTG', 'TTT']   

df_data[df_data['hgvs_nt'].str.contains('X', regex=False)]

df_data = df_data[~df_data.hgvs_nt.str.contains('X', regex=False)]

reference_sequence = 'GACGTTCCACTGCCGGCTGGTTGGGAAATGGCTAAAACTAGTTCTGGTCAGCGTTACTTCCTGAACCACATCGACCAGACCACCACGTGGCAGGACCCGCGT'
reference_list = list(reference_sequence)
site_start = 1
site_end = 34
site_list = list(range(site_start, site_end+1))

timepoints = [0, 1, 2, 3]

codon_length = 3

raw_codon = [reference_sequence[i:i+codon_length] for i in range(0, len(reference_sequence), codon_length)]

reps = [1, 2]
allele_counts_table = [df_data[['hgvs_nt','101208_c_0', '101208_c_1', '101208_c_2', '101208_c_3']], 
                       df_data[['hgvs_nt','110307_c_0', '110307_c_1', '110307_c_2', '110307_c_3']]]
#print(allele_counts_table[1])

allele_counts_list = []

for rep in reps:
    allele_counts_list = []
    rep_i = reps.index(rep)
    
    allele_counts_table_rep = allele_counts_table[rep_i]

    count_table_columns = allele_counts_table_rep.columns.tolist()

    allele_counts_table_no_wt = allele_counts_table_rep.drop(allele_counts_table_rep.index[allele_counts_table_rep['hgvs_nt'] == '_wt'])

    allele_counts_columns = ['replicate', 'generation', 'variants', 'counts']
    
    wt_number = allele_counts_table_rep[allele_counts_table_rep['hgvs_nt'] == '_wt'][count_table_columns[1:]]
    total_number = allele_counts_table_rep[allele_counts_table_rep.columns[1:]].astype(int).sum()
    
    for timepoint in timepoints:
        allele_counts_list.append([rep, timepoint, 'total_reads', total_number.tolist()[timepoints.index(timepoint)]])
        
    #print(wt_number.iloc[0].values.astype(int))
    for i in range(allele_counts_table_no_wt.shape[0]):
    #for i in range(10):
        #print("Progress {:2.1%}".format(i / allele_counts_table_no_wt.shape[0]), end="\r")
        variants_allele = allele_counts_table_no_wt.iloc[i].hgvs_nt
        mutation_number = allele_counts_table_no_wt.iloc[i].tolist()[1:]
        temp = [int(integer) for integer in mutation_number]
        mutation_number = temp
        #print(mutation_number)
        nucleotide = [x for x in variants_allele if x.isalpha()]
        variant_site = re.findall("(\d+)", variants_allele)

        nucleotide = nucleotide[1:]
        #print(nucleotide, variant_site)
        variant_list = reference_list.copy()
        for j in range(len(variant_site)):
            variant_list[int(variant_site[j])-1] = nucleotide[2 * j + 1]
        variant_sequence = ''.join(variant_list)
        variant_codon = [variant_sequence[i:i+codon_length] for i in range(0, len(variant_sequence), codon_length)]
        variant_codon_site = sorted(list(set([int((int(x)-1)/codon_length) for x in variant_site])))
        variant_record = [str(a+1)+b for a,b in zip(variant_codon_site, [variant_codon[i] for i in variant_codon_site])]
        #print(variant_record)
        variant_record = " ".join(variant_record)
        #print(variant_record)

        for timepoint in timepoints:
            allele_counts_list.append([rep, timepoint, variant_record, mutation_number[timepoints.index(timepoint)]])

    codon_counts_table = pd.DataFrame(data = allele_counts_list, columns = allele_counts_columns)
    codon_counts_table.to_csv('../outputs/epistasis/YAP1_genotype_count_rep%s.csv'%rep, sep = ',', index = False)

