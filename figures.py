import os
import sys

import numpy as np

import scipy as sp
import scipy.stats as st

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties
import matplotlib.patches as mpatches



import seaborn as sns 

import logomaker as lm

from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

# Plot variables
TEXT_FONTSIZE = 6
SERIAL_FONTSIZE = 8
TEXT_FONT = 'Arial'
FIG_DPI = 400
CM = 1/2.54

SERIAL_FONT = {
    'family': TEXT_FONT,
    'size': SERIAL_FONTSIZE,
    'weight': 'bold'
}

FONT = {
    'family': TEXT_FONT,
    'weight': 'normal',
    'size': TEXT_FONTSIZE
}

REGION_FONT = {
    'family': TEXT_FONT,
    'weight': 'bold',
    'size': TEXT_FONTSIZE
}

matplotlib.rc('font', **FONT)

FIG_FILE = './figures/'



# FIGURE 1 FINITE SAMPLING SIMULATION
# FIGURE 1 ARGUMENTS
matplotlib.rc_file_defaults()
matplotlib.rc('text', usetex=False)
FIG1_A_POS = {
    'x': 0.05,
    'y': 0.95
}
FIG1_B_POS = {
    'x': 0.05,
    'y': 0.48
}
FIG1_SIZE_X = 9
FIG1_SIZE_Y = 12
FIG1_SIMU_SCALE = 'linear'
FIG1_VAR          = 'F'
FIG1_HSPACE       = 0.4
FIG1_FINITE_NUM   = 10
FIG1_MARKER_SIZE  = 5
FIG1_GEN          = 10
FIG1_ALPHA        = 0.5
FIG1_BOX_ARG      = dict(left = 0.2, right = 0.95, bottom = 0.1, top = 0.95)
FIG1_FINITE_COLOR = 'grey'
FIG1_TRUE_COLOR   = 'red'
FIG1_TRUE_ALPHA   = 0.5

FIG1_TRAJECTORY_DIR = './outputs/simulation/WF_finite_sampling/'
FIG1_INFERENCE_DIR = './outputs/simulation/WF_mutational_effects/'
FIG1_WF_SIMU_FILE = './outputs/simulation/WF_simulation.csv'
FIG1_WF_SELECTION = FIG1_INFERENCE_DIR + '/selection_coefficients/'
FIG1_WF_LOGREG    = FIG1_INFERENCE_DIR+'/log_regression/'
FIG1_WF_RATIO     = FIG1_INFERENCE_DIR+'/enrichment_ratio/'
FIG1_WF_LOGRATIO  = FIG1_INFERENCE_DIR+'/enrichment_ratio_log/'
FIG1_FINITE_SIZE  = '_sampling-50000/'
FIG1_NAME         = 'Fig1_simulation.pdf'

# FIGURE 1 MAIN PLOT FUNCTION
def FIG1_SIMULATION_FINITE_SAMPLING():
    fig = plt.figure(figsize = (FIG1_SIZE_X * CM,FIG1_SIZE_Y * CM))
    gs  = fig.add_gridspec(2, 1, hspace = FIG1_HSPACE, **FIG1_BOX_ARG)

    fig.text(**FIG1_A_POS, s = 'a', **SERIAL_FONT, transform = fig.transFigure)
    fig.text(**FIG1_B_POS, s = 'b', **SERIAL_FONT, transform = fig.transFigure)

    # Plot trajectories
    ax1 = fig.add_subplot(gs[:1, 0])
    ax1.set_xlabel('Generation', fontsize = TEXT_FONTSIZE)
    ax1.set_ylabel('Allele frequency', fontsize = TEXT_FONTSIZE)
    ax1.tick_params(axis = 'both', which = 'major', labelsize = TEXT_FONTSIZE)
    ax1.set_yscale(FIG1_SIMU_SCALE)
    num_traj = 0



    for entry in os.scandir(FIG1_TRAJECTORY_DIR + 'gen-' + str(FIG1_GEN) + FIG1_FINITE_SIZE):
        if num_traj < 10:
            num_traj += 1
            if entry.path.endswith(".csv") and entry.is_file():
                df_trajectory = pd.read_csv(entry.path, index_col = 0)
                col_list = df_trajectory.columns.tolist()
                col_list.remove('generation')
                for i in [FIG1_VAR]:
                    ax1.plot(df_trajectory['generation'], df_trajectory[i], color = FIG1_FINITE_COLOR, alpha = FIG1_ALPHA)
        else:
            continue

    df_trajectory = pd.read_csv(FIG1_WF_SIMU_FILE, index_col=0)
    col_list      = df_trajectory.T.columns.tolist()
    for i in [FIG1_VAR]:
        ax1.plot(df_trajectory.columns[:FIG1_GEN], df_trajectory.T[i][:FIG1_GEN], color = FIG1_TRUE_COLOR, alpha = FIG1_TRUE_ALPHA)

    legend_fig1_a = [Line2D([0], [0], color = FIG1_FINITE_COLOR, lw = 2, label = 'Finite sampling'),
                     Line2D([0], [0], color = FIG1_TRUE_COLOR,   lw = 2, label = 'True evolution')]
    ax1.legend(handles = legend_fig1_a, frameon = False, fontsize = TEXT_FONTSIZE)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    ax2 = fig.add_subplot(gs[1:, 0])
    generation   = 10
    replicates   = 100
    sample_index = 10

    generations=[1, 4, 9, 19]
    finite_list = [50000]
    replicates = 100
    marker = ['o', 'v', '*']
    generation_ = [i + 1 for i in generations[: -1]]

    for finite_sampling in finite_list:
        temp = [[],[],[],[]]
        for generation in generation_:

            df_select               = pd.read_csv(FIG1_WF_SELECTION     + 'gen-%s_' %generation + 'sampling-%s' %finite_sampling + '.csv', index_col = 0)
            df_enrichment_regress   = pd.read_csv(FIG1_WF_LOGREG        + 'gen-%s_' %generation + 'sampling-%s' %finite_sampling + '.csv', index_col = 0)
            df_enrichment_ratio     = pd.read_csv(FIG1_WF_RATIO         + 'gen-%s_' %generation + 'sampling-%s' %finite_sampling + '.csv', index_col = 0)
            df_enrichment_ratio_log = pd.read_csv(FIG1_WF_LOGRATIO      + 'gen-%s_' %generation + 'sampling-%s' %finite_sampling + '.csv', index_col = 0)

            enrichment_ratio_corr       =     df_enrichment_ratio[    df_enrichment_ratio.columns[1:]].T.corr(method = 'pearson')
            log_regression_corr         =   df_enrichment_regress[  df_enrichment_regress.columns[1:]].T.corr(method = 'pearson')
            selection_coefficients_corr =               df_select[              df_select.columns[1:]].T.corr(method = 'pearson')
            enrichment_ratio_log_corr   = df_enrichment_ratio_log[df_enrichment_ratio_log.columns[1:]].T.corr(method = 'pearson')
            
            factor = replicates * replicates - replicates
            enrichment_ratio_corr       = (enrichment_ratio_corr.sum().sum()       - replicates)/factor
            log_regression_corr         = (log_regression_corr.sum().sum()         - replicates)/factor
            selection_coefficients_corr = (selection_coefficients_corr.sum().sum() - replicates)/factor  
            enrichment_ratio_log_corr   = (enrichment_ratio_log_corr.sum().sum()   - replicates)/factor
          
            enrichment_ratio_corr       =     df_enrichment_ratio.T.corr(method = 'pearson')
            log_regression_corr         =   df_enrichment_regress.T.corr(method = 'pearson')
            selection_coefficients_corr =               df_select.T.corr(method = 'pearson')
            enrichment_ratio_log_corr   = df_enrichment_ratio_log.T.corr(method = 'pearson')
            
            enrichment_ratio_corr = (enrichment_ratio_corr.sum().sum()             - replicates)/factor
            log_regression_corr = (log_regression_corr.sum().sum()                 - replicates)/factor
            selection_coefficients_corr = (selection_coefficients_corr.sum().sum() - replicates)/factor  
            enrichment_ratio_log_corr = (enrichment_ratio_log_corr.sum().sum()     - replicates)/factor
            
            temp[0].append(enrichment_ratio_corr)
            temp[1].append(log_regression_corr)
            temp[2].append(selection_coefficients_corr)
            temp[3].append(enrichment_ratio_log_corr)

        # ax2.plot(generation_, temp[0], c = 'red',    marker = marker[finite_list.index(finite_sampling)], markersize = FIG1_MARKER_SIZE, markeredgewidth = 0, alpha = 0.6)      
        # ax2.plot(generation_, temp[1], c = 'green',  marker  = marker[finite_list.index(finite_sampling)], markersize = FIG1_MARKER_SIZE, markeredgewidth = 0, alpha = 0.6)      
        ax2.plot(generation_, temp[2], c = '#4F94CD', marker = marker[finite_list.index(finite_sampling)], markersize = FIG1_MARKER_SIZE, markeredgewidth = 0, alpha = 0.6)      
        ax2.plot(generation_, temp[3], c = 'orange', marker  = marker[finite_list.index(finite_sampling)], markersize = FIG1_MARKER_SIZE, markeredgewidth = 0, alpha = 0.6)      
        ax2.set_xlabel('Generation used for inference',      fontsize = TEXT_FONTSIZE)
        ax2.set_ylabel('Pearson correlation \ncoefficients', fontsize = TEXT_FONTSIZE)
        ax2.set_xticks([2, 5, 10])
        ax2.set_xlim  ([   0, 11])
        ax2.set_ylim  ([-0.05, 1])
        
    legend_fig_1b = [Line2D([0], [0], color = '#4F94CD', lw = 2, label = 'Selection coefficients', alpha = 0.6),
                     # Line2D([0], [0], color = 'red',     lw = 2, label = 'Enrichment ratio',       alpha = 0.6),
                     Line2D([0], [0], color = 'orange',  lw = 2, label = 'Log scaled ratio',       alpha = 0.6),
                     # Line2D([0], [0], color = 'green',   lw = 2, label = 'Log regression',         alpha = 0.6)
                    ]

    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax2.set_xlim([1, 11])
    ax2.spines['right'].set_visible(False)
    ax2.spines[ 'top' ].set_visible(False)
    ax2.tick_params(axis = 'both', which = 'major', labelsize = TEXT_FONTSIZE)
    ax2.legend(handles = legend_fig_1b, frameon = False, fontsize = TEXT_FONTSIZE)
    fig.show()
    fig.savefig(FIG_FILE + FIG1_NAME, dpi = FIG_DPI)



######################


# FIGURE 2 METHODS COMPARISON 
# FIGURE 2 ARGUMENTS
matplotlib.rc_file_defaults()
matplotlib.rc('text', usetex=False)
FIG2_SIZE_X         = 18
FIG2_SIZE_Y         = 8
FIG2_A_MPL_COLOR    = '#4F94CD'#'blue'
FIG2_A_MPL_MARKER   = '*'
FIG2_A_PREF_COLOR   = '#FFA54F'#'orange'
FIG2_A_PREF_MARKER  = '^'
FIG2_A_MARKER_SIZE  = 20
FIG2_A_LEGEND_XY    = [0.05, 1.05]
FIG2_A_VIRUSTAG_XY  = [0.05, 0.35]
FIG2_A_HUMANTAG_XY  = [16.0, 0.35]
FIG2_A_TAGBOX       = dict(boxstyle='round', facecolor = 'white')
FIG2_VIRUS_DIR      = './outputs/virus_protein/'
FIG2_HUMAN_DIR      = './outputs/human_protein/'
FIG2_VIRUS_PREF_DIR = './data/virus_protein/'
FIG2_HUMAN_PREF_DIR = './data/human_protein/'

FIG2_A_VIRUS_RESULT_DIR = {
                   'Flu_A549':        ['PB2/A549/', 3],
                   'Flu_CCL141':      ['PB2/CCL141/', 3],
                   'Flu_Aichi68C':    ['Aichi68C_PR8/Aichi68C/', 2],
                   'Flu_PR8':         ['Aichi68C_PR8/PR8/', 2],
                   'Flu_MatrixM1':    ['Matrix_M1/', 3],
                   'Flu_MS':          ['MxA/MS/', 2],
                   'Flu_MxA':         ['MxA/MxA/', 2],
                   'Flu_MxAneg':      ['MxA/MxAneg/', 2],
                   'HIV_BG505':       ['HIVEnv/BG505/', 3],
                   'HIV_BF520':       ['HIVEnv/BF520/', 3],
                   'HIV_CD4_human':   ['HIVEnv_CD4/BF520_human/', 2],
                   'HIV_CD4_rhesus':  ['HIVEnv_CD4/BF520_rhesus/', 2],
                   'HIV_bnAbs_FP16':  ['HIV_bnAbs/FP16/', 2],
                   'HIV_bnAbs_FP20':  ['HIV_bnAbs/FP20/', 2],
                   'HIV_bnAbs_VRC34': ['HIV_bnAbs/VRC34/', 2]
                   }
FIG2_A_HUMAN_RESULT_DIR = {   
                   'HDR_Y2H_1':      ['BRCA1/Y2H_1/', 3],
                   'HDR_Y2H_2':      ['BRCA1/Y2H_2/', 3],
                   'HDR_E3':         ['BRCA1/E3/', 6],
                   'WWdomain_YAP1':  ['YAP1/', 2],
                   'Ubiq_Ube4b':     ['Ube4b/', 2],
                   'HDR_DRB1':       ['DRB1/', 2],
                   'Thrombo_TpoR_1': ['TpoR/TpoR_MPL/', 6],
                   'Thrombo_TpoR_2': ['TpoR/TpoR_S505NMPL/', 6]
                   }

FIG2_BOX_ARG         = dict(left = 0.12, right = 0.95, bottom = 0.3, top = 0.85)
FIG2_B_REPLICATE_NUM = 3
FIG2_B_SAMPLE_DIR    = './outputs/virus_protein/HIVEnv/BF520/selection_coefficients/'
FIG2_B_SCATTER_SIZE  = 7
FIG2_B_SCATTER_ALPHA = 0.4
FIG2_B_SCATTER_COLOR = 'orange'
FIG2_B_CORR_DIGIT    = 2
FIG2_NAME            = 'Fig2_comparison.pdf'

# FIGURE 2 MAIN PLOT FUNCTION
def FIG2_METHODS_COMPARISON():
    FIG2_A_PREF_AVG = {}
    FIG2_A_MPL_AVG  = {}
    for target_protein, info in FIG2_A_VIRUS_RESULT_DIR.items():
        path = FIG2_VIRUS_DIR+  info[0] + 'selection_coefficients/'
        for file in os.listdir(path):
            if file.endswith('.csv.gz'):
                df_temp = pd.read_csv(path + file)
                df_temp = df_temp[(df_temp['rep_1'] != 0) & (df_temp['rep_2'] != 0)]
                df_corr = df_temp[df_temp.columns[2:]]
                correlation_average = (df_corr.corr().sum().sum() - df_corr.shape[1])/(df_corr.shape[1]**2 - df_corr.shape[1])
                FIG2_A_MPL_AVG[target_protein] = correlation_average
                
    for protein, info_list in FIG2_A_VIRUS_RESULT_DIR.items():
        SELECTION_LIST = []
        ENRICH_LIST    = []
        replicate      = info_list[1]
        for file in os.listdir(FIG2_VIRUS_PREF_DIR + info_list[0] + 'pref'):
            if '.DS_Store' in file:
                continue
            else:
                FILE_PATH = FIG2_VIRUS_PREF_DIR + info_list[0] + 'pref/' + file
                temp_df   = pd.read_csv(FILE_PATH, index_col = 0)
                ENRICH_LIST.append(list(temp_df.values.flatten()))
            
        correlation_average = 0
        for i in range(len(ENRICH_LIST)):
            for j in range(i + 1, len(ENRICH_LIST)):
                correlation_average += st.pearsonr(ENRICH_LIST[i], ENRICH_LIST[j])[0]
        if correlation_average == 0:
            FIG2_A_PREF_AVG[protein] = correlation_average
        else:  
            correlation_average /= len(ENRICH_LIST)*(len(ENRICH_LIST) - 1)/2
            FIG2_A_PREF_AVG[protein] = correlation_average
        if 'HIV_bnAbs' in protein or 'Flu_MS' in protein or 'Flu_Mx' in protein:
            FILE_PATH = FIG2_VIRUS_PREF_DIR+info_list[0] + 'pref/enrichment.csv.gz'
            temp_df   = pd.read_csv(FILE_PATH)
            correlation_average      = (temp_df.corr().sum().sum() - temp_df.shape[1])/(temp_df.shape[1]**2 - temp_df.shape[1])
            FIG2_A_PREF_AVG[protein] = correlation_average

    PERFORMANCE_FIG_SIZE = (FIG2_SIZE_X * CM, FIG2_SIZE_Y * CM)
    fig = plt.figure(figsize = PERFORMANCE_FIG_SIZE)
    fig.text(0.05, 0.92, s = 'a', **SERIAL_FONT, transform = fig.transFigure)
    fig.text(0.6,  0.92, s = 'b', **SERIAL_FONT, transform = fig.transFigure)
    gs  = fig.add_gridspec(2, 5, wspace = 1.3, **FIG2_BOX_ARG)
    ax  = fig.add_subplot(gs[:2,:3])
    ax2 = fig.add_subplot(gs[:2,3:])
    FIG2_B_INNER = gridspec.GridSpecFromSubplotSpec(2, 2,
                                                    hspace = 0.6,
                                                    wspace = 0.5,
                                                    subplot_spec  = ax2, 
                                                    height_ratios = [1, 1], width_ratios = [1, 1])

    ax.set_xlabel('Target protein', fontsize = TEXT_FONTSIZE)
    ax.set_ylabel('Correlation coefficient of inference \n across replicates', fontsize = TEXT_FONTSIZE)

    MPL_LIST = FIG2_A_MPL_AVG.items()
    MPL_LIST = sorted(MPL_LIST, key = lambda x: x[1], reverse = True)
    x, y = zip(*MPL_LIST)
    ax.scatter(x,  y,  marker = FIG2_A_MPL_MARKER,  color = FIG2_A_MPL_COLOR,  s = FIG2_A_MARKER_SIZE)
    ENRICH_LIST = FIG2_A_PREF_AVG.items()
    x_, y_ = zip(*ENRICH_LIST)
    ax.scatter(x_, y_, marker = FIG2_A_PREF_MARKER, color = FIG2_A_PREF_COLOR, s = FIG2_A_MARKER_SIZE)
    labels = x
    ax.set_xticklabels(labels, rotation = 45, ha = 'right')

    # human
    FIG2_A_PREF_AVG = {}
    FIG2_A_MPL_AVG = {}
    FIG2_A_MPL_AVG[' '] = 100
    for target_protein, info in FIG2_A_HUMAN_RESULT_DIR.items():
        path = FIG2_HUMAN_DIR+info[0] + 'selection_coefficients/'
        for file in os.listdir(path):
            if file.endswith('.csv.gz'):
                df_temp = pd.read_csv(path + file)
                df_temp = df_temp[(df_temp['rep_1'] != 0)&(df_temp['rep_2'] != 0)]
                df_corr = df_temp[df_temp.columns[2:]]
                correlation_average = (df_corr.corr().sum().sum() - df_corr.shape[1])/(df_corr.shape[1]**2 - df_corr.shape[1])
                FIG2_A_MPL_AVG[target_protein] = correlation_average

    for protein, info_list in FIG2_A_HUMAN_RESULT_DIR.items():
        SELECTION_LIST = []
        ENRICH_LIST    = []
        replicate      = info_list[1]
        
        for file in os.listdir(FIG2_HUMAN_PREF_DIR + info_list[0] + 'pref'):
            FILE_PATH = FIG2_HUMAN_PREF_DIR + info_list[0] + 'pref/' + file
            if 'enrichment.csv.gz' in FILE_PATH:
                temp_df = pd.read_csv(FILE_PATH, index_col = 0)
                if 'hgvs_pro' in temp_df.columns:
                    temp_df = temp_df[~temp_df['hgvs_pro'].str.contains('\[')]
                df_corr = temp_df[temp_df.columns.tolist()[-info_list[1]:]]
                df_corr = df_corr.dropna()
                correlation_average = (df_corr.corr().sum().sum() - df_corr.shape[1])/(df_corr.shape[1]**2 - df_corr.shape[1])
                FIG2_A_PREF_AVG[protein] = correlation_average
 
    MPL_LIST = FIG2_A_MPL_AVG.items()
    MPL_LIST = sorted(MPL_LIST, key=lambda x: x[1], reverse = True)
    x, y = zip(*MPL_LIST)
    ax.scatter(x, y, marker = FIG2_A_MPL_MARKER, color = FIG2_A_MPL_COLOR, s = FIG2_A_MARKER_SIZE)

    ENRICH_LIST = FIG2_A_PREF_AVG.items()
    x_, y_ = zip(*ENRICH_LIST)
    ax.scatter(x_, y_, marker = FIG2_A_PREF_MARKER, color = FIG2_A_PREF_COLOR, s = FIG2_A_MARKER_SIZE)
    labels = labels + x
    ax.set_xticklabels(labels, rotation = 45, ha = 'right')
    ax.set_ylim(top = 1.05, bottom = 0.3)
    ax.axvline( x = ' ', ls = '--', color = 'black', lw = 1)
    legend_elements = [
                        Line2D([0], [0], marker = FIG2_A_MPL_MARKER, linestyle = 'None',
                               color = FIG2_A_MPL_COLOR,  label = 'MPL', markersize = 5),
                        Line2D([0], [0], marker = FIG2_A_PREF_MARKER, linestyle = 'None',
                               color = FIG2_A_PREF_COLOR, label = 'Ratio methods', markersize = 5)
                       ]
    ax.legend(handles = legend_elements, 
              loc = FIG2_A_LEGEND_XY, ncol = 2, 
              frameon = False, fontsize = TEXT_FONTSIZE, 
              handletextpad = 0.1)
    ax.spines['right'].set_visible(False)
    ax.spines[ 'top' ].set_visible(False)
    ax.tick_params(axis = 'both', which = 'major', labelsize = TEXT_FONTSIZE)
    ax.text(FIG2_A_VIRUSTAG_XY[0], FIG2_A_VIRUSTAG_XY[1], 'Virus', fontsize = TEXT_FONTSIZE, bbox=FIG2_A_TAGBOX)
    ax.text(FIG2_A_HUMANTAG_XY[0], FIG2_A_HUMANTAG_XY[1], 'Human', fontsize = TEXT_FONTSIZE, bbox=FIG2_A_TAGBOX)

    # Scatter plot of sample experiment
    SELECTION_LIST = []
    for file in os.listdir(FIG2_B_SAMPLE_DIR):
        if file.endswith('.csv.gz'):
            df_temp = pd.read_csv(FIG2_B_SAMPLE_DIR+file)
            df_temp = df_temp[(df_temp['rep_1'] != 0)&(df_temp['rep_2'] != 0)]
            for rep in range(1, FIG2_B_REPLICATE_NUM + 1):
                SELECTION_LIST.append(df_temp['rep_' + str(rep)].tolist())
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.spines[ 'top'  ].set_visible(False)
    ax2.spines['right' ].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines[ 'left' ].set_visible(False)

    SCATTER_DOT = {
        'alpha': 0.2,
        'edgecolor': 'none',
        's': FIG2_B_SCATTER_SIZE
    }

    for i in range(3):
        for j in range(i + 1, 3):
            CORR = round(st.pearsonr(SELECTION_LIST[i], SELECTION_LIST[j])[0], FIG2_B_CORR_DIGIT)
            if i == 0 and j == 1:
                ax2_sub = plt.Subplot(fig, FIG2_B_INNER[0, 0])
                ax2_sub.scatter(SELECTION_LIST[i], SELECTION_LIST[j], **SCATTER_DOT)
                ax2_sub.set_ylabel('Rep #2', fontsize = TEXT_FONTSIZE)

            if i == 0 and j == 2:
                ax2_sub = plt.Subplot(fig, FIG2_B_INNER[1, 0])
                ax2_sub.scatter(SELECTION_LIST[i], SELECTION_LIST[j], **SCATTER_DOT)
                ax2_sub.set_ylabel('Rep #3', fontsize = TEXT_FONTSIZE)
                ax2_sub.set_xlabel('Rep #1', fontsize = TEXT_FONTSIZE)

            if i == 1 and j == 2:
                ax2_sub = plt.Subplot(fig, FIG2_B_INNER[1, 1])
                ax2_sub.scatter(SELECTION_LIST[i], SELECTION_LIST[j], **SCATTER_DOT)
                ax2_sub.set_xlabel('Rep #2', fontsize = TEXT_FONTSIZE)
                
            ax2_sub.text(-0.8, 1.1, 'R = %.2f' %CORR, fontsize = TEXT_FONTSIZE)
            ax2_sub.set_yticks([-1, 0, 1])
            ax2_sub.set_xticks([-1, 0, 1])
            ax2_sub.set_xlim([-1.1, 1.3])
            ax2_sub.set_ylim([-1.1, 1.3])
            fig.add_subplot(ax2_sub)
            ax2_sub.spines['right'].set_visible(False)
            ax2_sub.spines[ 'top' ].set_visible(False)
            ax2_sub.tick_params(axis = 'both', which = 'major', labelsize = TEXT_FONTSIZE)
            ax2_sub.set_aspect('equal')
    ax2.set_xlabel('Selection coefficient', fontsize = TEXT_FONTSIZE, labelpad = 32) 
    ax2.set_ylabel('Selection coefficient', fontsize = TEXT_FONTSIZE, labelpad = 32) 
    fig.savefig(FIG_FILE + FIG2_NAME, dpi = 400)


#################
# FIGURE 3 VISUALIZATION 
# FIGURE 3 ARGUMENTS
matplotlib.rc_file_defaults()
matplotlib.rcParams.update({'font.size': TEXT_FONTSIZE})
SERIAL_FONT = {
    'size'  : TEXT_FONTSIZE + 2,
    'weight': 'bold'}
OFFSET_LETTER  = 0
SELECTION_FILE = FIG2_HUMAN_DIR + 'YAP1/selection_coefficients/YAP1_-4.csv.gz'
EPISTASIS_FILE = './outputs/epistasis/YAP1_100.txt'
INDEX_FILE     = './outputs/epistasis/index_matrix.csv'
SEQUENCE       = "DVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPR"
PAPER_FIGURE_SIZE_X = 18
PAPER_FIGURE_SIZE_Y = 14
EXAMPLE_FIG_SIZE    = (PAPER_FIGURE_SIZE_X * CM, PAPER_FIGURE_SIZE_Y * CM)

AA  = sorted(['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '*'])
Amino_acid_dict = {'Ala': 'A',
                   'Arg': 'R',
                   'Asn': 'N',
                   'Asp': 'D',
                   'Cys': 'C',
                   'Gln': 'Q',
                   'Glu': 'E',
                   'Gly': 'G',
                   'His': 'H',
                   'Ile': 'I',
                   'Leu': 'L',
                   'Lys': 'K',
                   'Met': 'M',
                   'Phe': 'F',
                   'Pro': 'P',
                   'Ser': 'S',
                   'Thr': 'T',
                   'Trp': 'W',
                   'Tyr': 'Y',
                   'Val': 'V',
                   'Ter': '*'
                  }

def FIG3_VISUALIZATION(exp_scale = 10, sites_per_line = 35):
    
    fig = plt.figure(figsize = EXAMPLE_FIG_SIZE)
    text_in_figure = {
        'fontsize': TEXT_FONTSIZE,
        'bbox'    : {'facecolor': 'none', 'edgecolor': 'none', 'boxstyle': 'round'},
        'ha'      : 'center',
        'va'      : 'center'
    }
    fig.text(0.02, 0.94, s = 'a', **SERIAL_FONT, transform = fig.transFigure)
    fig.text(0.47, 0.94, s = 'b', **SERIAL_FONT, transform = fig.transFigure)
    fig.text(0.02, 0.52, s = 'c', **SERIAL_FONT, transform = fig.transFigure)
    fig.text(0.47, 0.52, s = 'd', **SERIAL_FONT, transform = fig.transFigure)
    data = pd.read_csv(SELECTION_FILE)
    site_list = data['site'].unique().tolist()
    rep_list  = data.columns[2:]
    BOX_FIGURE3 = dict(left = 0.065, right = 1, bottom = 0.1, top = 0.9)
    gs = fig.add_gridspec(5, 6, 
                          wspace = 1, 
                          hspace = 3.5, 
                          width_ratios  = [1,   1,   1,   1.6, 1.6, 1.6], 
                          height_ratios = [1.2, 1.2, 1.6, 1.6, 1.6], **BOX_FIGURE3)
    ax1 = fig.add_subplot(gs[:2, :3])
    ax2 = fig.add_subplot(gs[:2, 3:])
    ax3 = fig.add_subplot(gs[2:, :3])
    ax4 = fig.add_subplot(gs[2:, 3:])
    MPL_scale  = 8
    PREF_scale = 2
    for rep in [rep_list[0]]:
# logo plot
        data1 = data[['site', 'amino_acid', rep]]
        data1 = pd.pivot_table(data1, values = rep, index = ['site'], columns = ['amino_acid']).reset_index()
        data1.set_index('site', inplace = True)
        data1 = data1.drop(['*'], axis = 1)
        data1.replace(0, np.nan, inplace = True)
        totle_line_num = len(data1.columns)
        num_line = int(totle_line_num/sites_per_line) + 1
        for i in range(num_line):
            if i != num_line - 1:
                data_sub = data1[i*sites_per_line: (i + 1) * sites_per_line]               
            else:
                data_sub = data1[i*sites_per_line:]
            site_sub = data_sub.columns.tolist()  
            data_exp = np.exp(MPL_scale*data_sub)
            data_exp = data_exp.div(data_exp.sum(axis = 1), axis = 0)            
            data_exp = data_exp.replace(np.nan, 0)
            logo = lm.Logo(data_exp, 
                           ax           = ax1,
                           figsize      = [15 * len(site_sub)/sites_per_line,4],
                           font_name    = 'Arial Rounded MT Bold',
                           color_scheme = 'dmslogo_funcgroup')
            
            logo.style_xticks( fmt = '%d', anchor = 0)
            logo.ax.set_ylabel("Normalized \nselection coefficients", fontsize = TEXT_FONTSIZE)
            logo.ax.set_xlabel("Site", fontsize = TEXT_FONTSIZE)
            
            for j in range(len(SEQUENCE)):
                logo.ax.text(j + 2, 1.05, SEQUENCE[j], **text_in_figure, color = 'grey')
            logo.ax.text(18.5, 1.125, 'Wild type sequence', **text_in_figure, color = 'grey') 
            logo.style_spines(spines=['bottom', 'right', 'left', 'top'], visible = False)
            
        ax1.set_yticks([])
        ax1.set_xticks([i +  data_exp.index.tolist()[0] for i in range(len(data_exp.index.tolist())) if (i)%5 == 0])
        ax1.set_xticklabels([data_exp.index.tolist()[i] + 8 for i in range(len(data_exp.index.tolist())) if (i)%5 == 0])
        ax1.tick_params(axis = u'both', which = u'both', length = 0)
        ax1.yaxis.labelpad = 10
        ax1.xaxis.set_tick_params(width = 0)

# heatmap
        left_n = 3
        data1 = data[['site', 'amino_acid', rep]].copy()
        data1 = pd.pivot_table(data1, values = rep, index = ['amino_acid'], columns = 'site').reset_index()
        data1.set_index('amino_acid', inplace = True)
        totle_line_num = len(data1.columns)
        num_line = int(totle_line_num/sites_per_line) + 1
        for i in range(num_line):
            if i != num_line - 1:
                data_sub = data1[data1.columns[i * sites_per_line: (i + 1) * sites_per_line]]               
            else:
                data_sub = data1[data1.columns[i * sites_per_line:]]
            site_sub = data_sub.columns.tolist()
            norm = mcolors.TwoSlopeNorm(vcenter = 0)
            color_map = plt.cm.get_cmap('RdBu')
            hm = ax2.imshow(data_sub,
                            cmap   = color_map.reversed(), 
                            norm   = norm, 
                            aspect = 'equal',
                            extent = (-0.5 - left_n, data_sub.shape[1] - 0.5 - left_n, data_sub.shape[0] - 0.5, -0.5))
            white_entry = data_sub.isin([0])
            return_df   = data_sub.copy()
            white_entry = white_entry.astype(int)
            white_entry = white_entry.iloc[::-1]
            cmap   = mcolors.ListedColormap(['#FF000000', 'lightgray'])
            bounds = [-1,0.5,1.5]
            norm   = mcolors.BoundaryNorm(bounds, cmap.N)
            ax2.imshow(white_entry, interpolation = 'nearest', origin = 'lower',
                       cmap = cmap, norm = norm,aspect = 'equal',
                       extent = (-0.5 - left_n, data_sub.shape[1] - 0.5 - left_n, data_sub.shape[0] - 0.5, -0.5))

            clb = fig.colorbar(hm, ax = ax2, orientation = 'vertical', pad = 0.1)
            clb.ax.set_xlabel('Selection\ncoefficient', fontsize = TEXT_FONTSIZE)
            index_of_element_to_outline = []

            for j in range(len(site_sub)):
                index_of_element_to_outline.append([j - 0.5 - left_n, AA.index(SEQUENCE[site_list.index(site_sub[j])]) - 0.5])
           
            for outliner in index_of_element_to_outline:
                ax2.scatter(outliner[0] + 0.5, outliner[1] + 0.5, c = 'black', s = 2)
                
        ax2.set_xlabel("Site", fontsize = TEXT_FONTSIZE)
        ax2.set_ylabel("Amino acid", fontsize = TEXT_FONTSIZE)
        ax2.set_xticks([i-3 for i in range(len(site_sub)) if i%5 == 0])
        ax2.set_xticklabels([site_sub[i]+8 for i in range(len(site_sub)) if (i)%5 == 0])
        ax2.set_yticks([i for i in range(len(AA))])
        ax2.set_yticklabels([AA[i] for i in range(len(AA))])
        ax2.tick_params(axis = u'both', which = u'both', length = 0)
        ax2.spines[ 'right'].set_visible(False)
        ax2.spines[  'top' ].set_visible(False)
        ax2.spines[ 'left' ].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        
        legend_elements = [
                            Line2D([], [], marker = 'o', linestyle = 'None',
                                  color = 'black', label = 'Wild type', markersize = 1.5),
                            Line2D([], [], marker = 's', linestyle = 'None',
                                  markerfacecolor = 'lightgray', markeredgewidth = 0, label = 'Missing variant', markersize = 5)
                           ]
        ax2.legend(handles = legend_elements, loc = [0.4, 1.05], ncol = 2, frameon = False, fontsize = TEXT_FONTSIZE, handletextpad = 0.1)
        ax2.set_position([0.52, 0.6, 0.37, 0.37])

# Comparison Logoplot
        inner = gridspec.GridSpecFromSubplotSpec(3, 1,subplot_spec = ax3, wspace = 0.1, hspace = 0.5)
        df_enrich   = pd.read_csv('./data/human_protein/YAP1/pref/enrichment.csv.gz')
        df_enrich['hgvs_pro'].astype(str)
        df_enrich   = df_enrich[~df_enrich['hgvs_pro'].str.contains('\[')]
        df_enrich   = df_enrich[ df_enrich['hgvs_pro'].str.contains('p.')]
        mutant_list = df_enrich['hgvs_pro'].tolist()
        pref_list   = df_enrich['score_101208'].tolist()
        AA_dict     = {}
        for aa in AA:
            if aa != '*':
                AA_dict[aa] = [0] * 34
        site_lll = set()
        for i in range(len(mutant_list)):
            if mutant_list[i][-1] != '?':
                mutants = mutant_list[i][-3:]
                site_lll.add(int(mutant_list[i][5: -3]))
                short_AA = Amino_acid_dict[mutants]
                if short_AA != '*':
                    if mutant_list[i][5:-3] != '':
                        site = int(mutant_list[i][5: -3]) - 1
                    AA_dict[short_AA][site] = pref_list[i]

        df_AA = pd.DataFrame(columns = ['site', 'amino_acid', 'pref'])
        site_list = []
        amino_acid_list = []
        pref_list = []
        for aa, pref in AA_dict.items():
            for i in range(len(pref)):
                site_list.append(i + 2)
                pref_list.append(pref[i])
                amino_acid_list.append(aa)
        df_AA['site'] = site_list
        df_AA['amino_acid'] = amino_acid_list
        df_AA['pref'] = pref_list
        data2 = pd.pivot_table(df_AA, values = 'pref', index = ['site'], columns = ['amino_acid']).reset_index()
        data2['site'] = [str(i) + '_Pref' for i in data2['site'].tolist()]

        data = pd.read_csv(SELECTION_FILE)
        site_list = data['site'].unique().tolist()
        rep_list = data.columns[2:]

        sites_per_line = 35
        exp_scale = 10
        plt.figure()
        data1 = data[['site', 'amino_acid', 'rep_1']]

        data1 = pd.pivot_table(data1, values = rep, index=['site'], columns = ['amino_acid']).reset_index()
        data1 = data1.drop(['*'], axis = 1)
        data1['site'] = [str(i) + '_MPL' for i in data1['site'].tolist()]


        df_all  = data1[0: 0].copy()
        df_zero = data1[0: 1].copy()
        for col in df_zero.columns:
            df_zero[col].values[:] = 0
        selected_rows = [10, 16, 27, 
                         4,  13, 28,
                         29, 30, 20]
        k = 10 * len(selected_rows)
        m = 0
        for i in selected_rows:
            #MPL
            if m%3 == 0:
                df_all = df_all.append(df_zero)
            m += 1
            df_all    = df_all.append(df_zero)
            df_temp   = data1[data1['site'] == str(i) + '_MPL'].copy()
            temp_pure = df_temp[df_temp.columns[1:]]
            
            temp_pure = np.exp(MPL_scale * temp_pure)
            temp_pure.replace(1, 0, inplace = True)
            temp_pure = temp_pure.div(temp_pure.sum(axis = 1), axis = 0)
            df_all = df_all.append(temp_pure)
            
            #PREF
            df_temp   = data2[data2['site'] == str(i) + '_Pref'].copy()
            temp_pure = df_temp[df_temp.columns[1:]]
            temp_pure = np.exp(PREF_scale * temp_pure)
            temp_pure.replace(1, 0, inplace = True)
            temp_pure = temp_pure.div(temp_pure.sum(axis = 1), axis = 0)
            df_all = df_all.append(temp_pure)
        
        df_all['plot_site'] = [i for i in range(10)] + [i for i in range(10)] + [i for i in range(10)]
        plot_index = df_all['site']
        df_all = df_all.drop(['site'], axis = 1)

        df_all.set_index('plot_site', inplace = True)
        total_line_num = len(df_all.columns)
        sites_per_line = int(df_all.shape[0]/3)
        num_line = int(total_line_num/sites_per_line) + 1
        ax3.set_yticks([])
        ax3.set_xticks([])
        
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.spines['left'].set_visible(False)

        for i in range(num_line):
            ax3_sub = plt.Subplot(fig, inner[i])
            if i != num_line -1:
                data_sub = df_all[i * sites_per_line: (i + 1) * sites_per_line].copy()              
            else:
                data_sub = df_all[i * sites_per_line:].copy()
            site_sub = data_sub.columns.tolist()  
            logo = lm.Logo(data_sub, 
                           ax           = ax3_sub,
                           figsize      = [5, 2],
                           font_name    = 'Arial Rounded MT Bold',
                           color_scheme = 'dmslogo_funcgroup')

            logo.style_spines(spines = ['right', 'top'], visible = False)
            logo.ax.set_xlim([-2, 10])

            if i == 0:
                logo.ax.text(2.5, 1.1, 'Site 10(E)', **text_in_figure)
                logo.ax.text(5.5, 1.1, 'Site 16(S)', **text_in_figure)
                logo.ax.text(8.5, 1.1, 'Site 27(Q)', **text_in_figure)
                logo.ax.text(-.5, .5, 'Similar\ninference\n(Wild Type)', **text_in_figure)
                
            if i == 1:
                logo.ax.text(2.5, 1.1, 'Site  4(P)', **text_in_figure)
                logo.ax.text(5.5, 1.1, 'Site 13(K)', **text_in_figure)
                logo.ax.text(8.5, 1.1, 'Site 28(T)', **text_in_figure)
                logo.ax.set_ylabel("Normalized measurements", fontsize = TEXT_FONTSIZE, labelpad=10)
                logo.ax.text(-.5, .5, 'Similar\ninference\n(Tolerance)', 
                             fontsize = TEXT_FONTSIZE, 
                             bbox=dict(facecolor = 'none', 
                                       edgecolor = 'none', 
                                       boxstyle  = 'round'), 
                             ha = 'center', va = 'center')

            else:
                logo.ax.set_ylabel(" ", fontsize = TEXT_FONTSIZE, labelpad = 3)
            if i == 2:
                logo.ax.text(2.5, 1.1, 'Site 29(T)', **text_in_figure)
                logo.ax.text(5.5, 1.1, 'Site 30(T)', **text_in_figure)
                logo.ax.text(8.5, 1.1, 'Site 34(P)', **text_in_figure)
                logo.ax.text(-.5, .5, 'Different\ninference', 
                             fontsize = TEXT_FONTSIZE, 
                             bbox = dict(facecolor ='none', 
                                       edgecolor   = 'none', 
                                       boxstyle    = 'round'), 
                             ha = 'center', va = 'center')

                logo.ax.set_xticks([2, 3, 4, 5, 6, 7, 8, 9])
                logo.ax.set_xticklabels(['MPL', 'PREF', ' ', 'MPL', 'PREF', ' ', 'MPL', 'PREF'], linespacing = 1.5, rotation = 30)
                logo.ax.xaxis.get_major_ticks()[2].set_visible(False)
                logo.ax.xaxis.get_major_ticks()[5].set_visible(False)
            else:
            
                logo.ax.set_xticks([i for i in range(2,10)])
                logo.ax.set_xticklabels([' '] * 8)
                logo.ax.xaxis.get_major_ticks()[2].set_visible(False)
                logo.ax.xaxis.get_major_ticks()[5].set_visible(False)
            logo.ax.set_yticks([])
            logo.ax.spines['left'].set_visible(False)
            fig.add_subplot(ax3_sub)      
        
# epistasis
        with open(EPISTASIS_FILE) as f:
            content = f.readlines()
        content = [float(x.strip()) for x in content]
        df_index = pd.read_csv(INDEX_FILE, header=None)
        df_index.rename(columns = {0 : 'state_1', 1 : 'state_2', 2 : 'mat_index'}, inplace = True)
        index_list = df_index['mat_index'].tolist()
        selection_list = [content[i] for i in index_list]
        df_index['selection_coefficients'] = selection_list
        df_index['site_1'] = df_index['state_1']/21 + 2
        df_index['site_1'] = df_index['site_1'].astype(int)
        df_index['AA_1']   = df_index['state_1']%21
        df_index['AA_1']   = [AA[x] for x in df_index['AA_1'].tolist()]
        df_index['site_2'] = df_index['state_2']/21 + 2
        df_index['site_2'] = df_index['site_2'].astype(int)
        df_index['AA_2']   = df_index['state_2']%21
        df_index['AA_2']   = [AA[x] for x in df_index['AA_2'].tolist()]

        cols = ['site_1', 'AA_1', 'site_2', 'AA_2']
        df_index['all_variant'] = df_index[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis = 1)

        WT_list = []; axis_1 = []; axis_2 = []
        for i in range(len(SEQUENCE)):
            WT_list.append(i * 21 + AA.index(SEQUENCE[i]))
        df_WT_epi = df_index[(~df_index['state_1'].isin(WT_list))&(~df_index['state_2'].isin(WT_list))&(df_index['state_2'] != df_index['state_1'])]
        df_WT_epi = df_WT_epi.sort_values(['site_1', 'site_2'])

        df_WT_epi['absolute_s'] = np.abs(df_WT_epi['selection_coefficients'])
        df_WT_epi = df_WT_epi.groupby(['site_1', 'site_2'], as_index=False)['absolute_s'].agg('sum')
        df_WT_epi['site_1'] += 8
        df_WT_epi['site_2'] += 8
        df_WT_epi_ = df_WT_epi.copy()
        df_WT_epi_ = df_WT_epi_.rename(columns={"site_1":"site_2", "site_2":"site_1"})
        df_WT_epi  = df_WT_epi.append(df_WT_epi_, ignore_index=True, sort=False)
        flights = df_WT_epi.pivot("site_1", "site_2", "absolute_s")
        hm = ax4.imshow(flights,cmap='Greys', aspect = 'equal', vmin = 0)
        ax4.set_xlabel('Site', labelpad = 7)
        ax4.set_ylabel('Site', labelpad = 7)
        ax4.spines['right'].set_visible(False)
        ax4.spines['top'].set_visible(False)
        ax4.spines['left'].set_visible(False)
        ax4.spines['bottom'].set_visible(False)

        ax4.set_xticks([i for i in range(1,len(SEQUENCE)) if (i)%5 == 0])
        ax4.set_xticklabels([i + 10 for i in range(1,len(SEQUENCE)) if (i)%5 == 0])
        ax4.set_yticks([i for i in range(1,len(SEQUENCE)) if (i)%5 == 0])
        ax4.set_yticklabels([i + 10 for i in range(1,len(SEQUENCE)) if (i)%5 == 0])
        ax4.tick_params(axis = u'both', which = u'both', length = 0)
        ax4.invert_yaxis()
        
        
        blue_alpha   = 0.5
        orange_alpha = 0.4
        
        ax4.axhspan(-0.5, 2.5, color = "lightblue", alpha = blue_alpha, lw = 0)
        ax4.axvspan(-0.5, 2.5, color = "lightblue", alpha = blue_alpha, lw = 0)
        
        
        ax4.axhspan(11.5, 16.5, color = "lightblue", alpha = blue_alpha, lw = 0)
        ax4.axvspan(11.5, 16.5, color = "lightblue", alpha = blue_alpha, lw = 0)

        
        ax4.axhspan(21.5, 25.5, color = "lightblue", alpha = blue_alpha, lw = 0)
        ax4.axvspan(21.5, 25.5, color = "lightblue", alpha = blue_alpha, lw = 0)
        
        ax4.fill([-0.5, -0.5, 2.5,  2.5],  [-0.5, 2.5,  2.5,  -0.5], 'coral', alpha = orange_alpha)
        ax4.fill([-0.5, -0.5, 2.5,  2.5],  [11.5, 16.5, 16.5, 11.5], 'coral', alpha = orange_alpha)
        ax4.fill([11.5, 16.5, 16.5, 11.5], [-0.5, -0.5, 2.5,  2.5],  'coral', alpha = orange_alpha)
        ax4.fill([11.5, 16.5, 16.5, 11.5], [11.5, 11.5, 16.5, 16.5], 'coral', alpha = orange_alpha)
        ax4.fill([21.5, 25.5, 25.5, 21.5], [-0.5, -0.5, 2.5,  2.5],  'coral', alpha = orange_alpha)
        ax4.fill([-0.5, -0.5, 2.5,  2.5],  [21.5, 25.5, 25.5, 21.5], 'coral', alpha = orange_alpha)
        ax4.fill([21.5, 21.5, 25.5, 25.5], [21.5, 25.5, 25.5, 21.5], 'coral', alpha = orange_alpha)
        ax4.fill([11.5, 16.5, 16.5, 11.5], [21.5, 21.5, 25.5, 25.5], 'coral', alpha = orange_alpha)
        ax4.fill([21.5, 21.5, 25.5, 25.5], [11.5, 16.5, 16.5, 11.5], 'coral', alpha = orange_alpha)

        ax4.axvline(x = 2.5,  ls = '--', color = 'black', lw = 1)
        ax4.axvline(x = 11.5, ls = '--', color = 'black', lw = 1)
        ax4.axvline(x = 16.5, ls = '--', color = 'black', lw = 1)
        ax4.axvline(x = 21.5, ls = '--', color = 'black', lw = 1)
        ax4.axvline(x = 25.5, ls = '--', color = 'black', lw = 1)
        
        ax4.axhline(y = 2.5,  ls = '--', color = 'black', lw = 1)
        ax4.axhline(y = 11.5, ls = '--', color = 'black', lw = 1)
        ax4.axhline(y = 16.5, ls = '--', color = 'black', lw = 1)
        ax4.axhline(y = 21.5, ls = '--', color = 'black', lw = 1)
        ax4.axhline(y = 25.5, ls = '--', color = 'black', lw = 1)

        clb = fig.colorbar(hm, ax = ax4, orientation = 'vertical', pad = 0.07, shrink = 0.6)
        clb.ax.set_xlabel(r'$\sum$|epistasis|', fontsize = TEXT_FONTSIZE)
        
    plt.tight_layout()
    fig.savefig(FIG_FILE + 'Fig3_realdata.pdf', dpi = 400)
    plt.show()  
    return flights



###
SUPPFIG1_RAWDATA_REP1 = './outputs/epistasis/tRNA/rep1_100.txt'
SUPPFIG1_IDXMAT_REP1  = './outputs/epistasis/tRNA/index_matrix_rep1.csv'
SUPPFIG1_RAWDATA_REP2 = './outputs/epistasis/tRNA/rep2_100.txt'
SUPPFIG1_IDXMAT_REP2  = './outputs/epistasis/tRNA/index_matrix_rep2.csv'

def SUPPFIG1_EPISTASIS():
    with open(SUPPFIG1_RAWDATA_REP1, 'r') as f:
        lines = f.readlines()
    epistasis_selection = []
    for line in lines:
        epistasis_selection.append(float(line))
        
    df_index1 = pd.read_csv(SUPPFIG1_IDXMAT_REP1, names = ['variant_1', 'variant_2', 'index'], header = None)
    start_site = df_index1['variant_1'].min()
    end_site = df_index1['variant_2'].max()
    df_i = []
    for i in range(end_site+1):
        df_i.append(df_index1[(df_index1['variant_1'] == i) & (df_index1['variant_2'] == i)].index.values[0])
    index_list = df_index1['index'].tolist()
    selection_list = []
    for index in index_list:
        selection_list.append(epistasis_selection[index])
    df_index1['selection_coefficient'] = selection_list

    with open(SUPPFIG1_RAWDATA_REP2, 'r') as f:
        lines = f.readlines()
    epistasis_selection = []
    for line in lines:
        epistasis_selection.append(float(line))
        
    df_index2 = pd.read_csv(SUPPFIG1_IDXMAT_REP2, names = ['variant_1', 'variant_2', 'index'], header = None)
    start_site = df_index2['variant_1'].min()
    end_site = df_index2['variant_2'].max()
    df_i = []
    for i in range(end_site+1):
        df_i.append(df_index2[(df_index2['variant_1'] == i) & (df_index2['variant_2'] == i)].index.values[0])
    index_list = df_index2['index'].tolist()
    selection_list = []
    for index in index_list:
        selection_list.append(epistasis_selection[index])
    df_index2['selection_coefficient'] = selection_list

    result = pd.merge(df_index1, df_index2, on=["variant_1", "variant_2"])

    print(result[["selection_coefficient_x", "selection_coefficient_y"]].corr(), result.shape)
    result.plot.scatter(x='selection_coefficient_x', y='selection_coefficient_y', alpha=0.1)
    plt.ylabel('rep #1')
    plt.xlabel('rep #2')
    plt.savefig('Sup_Fig1_epistasis_correlation.pdf', dpi=200)





