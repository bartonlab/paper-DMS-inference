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
import matplotlib.ticker as ticker

import mplot as mp

import seaborn as sns 

import logomaker as lm

from matplotlib.lines import Line2D


# Plot variables
FIG_DPI = 400

COLOR_1  = '#1295D8'
COLOR_2  = '#FFB511'
BKCOLOR  = '#252525'
LCOLOR   = '#969696'
C_BEN    = '#EB4025' #'#F16913'
C_BEN_LT = '#F08F78' #'#fdd0a2'
C_NEU    =  LCOLOR   #'#E8E8E8' # LCOLOR
C_NEU_LT = '#E8E8E8' #'#F0F0F0' #'#d9d9d9'
C_DEL    = '#3E8DCF' #'#604A7B'
C_DEL_LT = '#78B4E7' #'#dadaeb'
C_POP  = C_DEL
C_PREF = LCOLOR

cm2inch = lambda x: x/2.54
SINGLE_COLUMN   = cm2inch(8.8)
ONE_FIVE_COLUMN = cm2inch(11.4)
DOUBLE_COLUMN   = cm2inch(18.0)
SLIDE_WIDTH     = 10.5
GOLDR           = (1.0 + np.sqrt(5)) / 2.0

# paper style
FONTFAMILY    = 'Arial'
SIZESUBLABEL  = 8
SIZELABEL     = 6
SIZETICK      = 6
SMALLSIZEDOT  = 6.
SIZELINE      = 0.6
AXES_FONTSIZE = 6
AXWIDTH       = 0.4

DEF_SUBLABEL = {
    'family': FONTFAMILY,
    'size':   SIZESUBLABEL,
    'weight': 'bold'
}

DEF_ERRORPROPS = {
    'mew'        : AXWIDTH,
    'markersize' : SMALLSIZEDOT/2,
    'fmt'        : 'o',
    'elinewidth' : SIZELINE/2,
    'capthick'   : 0,
    'capsize'    : 0
}

DEF_LABELPROPS = {
    'family' : FONTFAMILY,
    'size'   : SIZELABEL,
    'color'  : BKCOLOR,
    'clip_on': False
}

DEF_TEXTPROPS = {
    'family' : FONTFAMILY,
    'size'   : SIZELABEL
}

FIGPROPS = {
    'transparent' : True,
    #'bbox_inches' : 'tight'
}

matplotlib.rc('font', **DEF_TEXTPROPS)

FIG_DIR  = './figures/'
PREF_DIR = './output/merged_preference/'
POP_DIR  = './output/selection_coefficients/'

COL_SITE = 'site'
COL_AA   = 'amino_acid'
COL_WT   = 'WT_indicator'
COL_S    = 'joint'

AA = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '*']

NAME2NAME = {
    'Flu_WSN':  'WSN',
    'Flu_A549': 'PB2 A549',
    'Flu_CCL141': 'PB2 CCL141',
    'HIV_BG505': 'Env BG505',
    'HIV_BF520': 'Env BF520',
    'HIV_CD4_human': 'BF520 hu',
    'HIV_CD4_rhesus': 'BF520 rhm',
    'ZIKV': 'ZIKV',
    'Perth2009': 'Perth2009',
    'HIV_bnAbs_VRC34': 'VRC34',
    'HIV_bnAbs_FP16': 'FP16-02',
    'HIV_bnAbs_FP20': 'FP20-01',
    'Flu_MS': 'NP MS',
    'Flu_MxA': 'NP MxA',
    'Flu_MxAneg': 'NP MxAneg',
    'Flu_MatrixM1': 'M1',
    'Flu_Aichi68C': 'Aichi68',
    'Flu_PR8': 'PR8',
    'WWdomain_YAP1': 'WW',
    'Ubiq_Ube4b': 'Ube4b',
    'HDR_Y2H_1': 'Y2H 1',
    'HDR_Y2H_2': 'Y2H 2',
    'HDR_E3': 'E3',
    'HDR_DBR1': 'DBR1',
    'Thrombo_TpoR_1': 'TpoR 1',
    'Thrombo_TpoR_2': 'TpoR 2',
}


######################
# FIGURE 1 OVERVIEW AND METHODS COMPARISON

def fig_methods_comparison():
    ''' FUTURE: PASS FIGURE NAME AND OTHER RELATIVE PARAMETERS INTO THE FUNCTION '''

    input_files = {'Flu_WSN':         ['WSN',                   3],#
                   'Flu_A549':        ['A549',                  2],#
                   'Flu_CCL141':      ['CCL141',                3],#
                   'Flu_Aichi68C':    ['Aichi68C',              2],#
                   'Flu_PR8':         ['PR8' ,                  2],#
                   'Flu_MatrixM1':    ['Matrix_M1',             3],#
                   'ZIKV':            ['ZIKV',                  3],
                   'Perth2009':       ['Perth2009',             4],
                   'Flu_MS':          ['MS',                    2],#
                   'Flu_MxA':         ['MxA',                   2],#
                   'Flu_MxAneg':      ['MxAneg',                2],#
                   'HIV_BG505':       ['HIV Env BG505' ,        3],#
                   'HIV_BF520':       ['HIV Env BF520' ,        3],#
                   'HIV_CD4_human':   ['HIV BF520 human host',  2],#
                   'HIV_CD4_rhesus':  ['HIV BF520 rhesus host', 2],#
                   'HIV_bnAbs_FP16':  ['HIV bnAbs FP16',        2],
                   'HIV_bnAbs_FP20':  ['HIV bnAbs FP20',        2],
                   'HIV_bnAbs_VRC34': ['HIV bnAbs VRC34',       2],
                   'HDR_Y2H_1':       ['Y2H_1',                 3],
                   'HDR_Y2H_2':       ['Y2H_2',                 3],
                   'HDR_E3':          ['E3',                    6],
                   'WWdomain_YAP1':   ['YAP1',                  2],
                   'Ubiq_Ube4b':      ['Ube4b',                 2],
                   'HDR_DBR1':        ['DBR1',                  2],
                   'Thrombo_TpoR_1':  ['TpoR',                  6],
                   'Thrombo_TpoR_2':  ['TpoR_S505N',            6]  }

    fig_title = 'fig-1-overview.pdf'

    pref_avg = {}
    pop_avg  = {}
    for target_protein, info in input_files.items():
        reps = info[1]
        rep_list = ['rep_'+str(i+1) for i in range(reps)]
        path = POP_DIR + info[0] + '.csv.gz'
        df_sele = pd.read_csv(path)        

        path = PREF_DIR +  info[0] + '.csv.gz'
        df_pref = pd.read_csv(path)

        df_merged = pd.merge(df_pref, df_sele, on=['site', 'amino_acid'])
        df_corr_pref = df_merged[[i+'_x' for i in rep_list]]
        df_corr_sele = df_merged[[i+'_y' for i in rep_list]]
        correlation_average = (df_corr_pref.corr().sum().sum() - df_corr_pref.shape[1])/(df_corr_pref.shape[1]**2 - df_corr_pref.shape[1])
        pref_avg[target_protein] = correlation_average
        correlation_average = (df_corr_sele.corr().sum().sum() - df_corr_sele.shape[1])/(df_corr_sele.shape[1]**2 - df_corr_sele.shape[1])
        pop_avg[target_protein] = correlation_average
    
    # variables
    w = DOUBLE_COLUMN
    h = DOUBLE_COLUMN * 1.1 / GOLDR

    fig = plt.figure(figsize = (w, h))
    
    box_t = 0.98
    box_b = 0.07
    box_l = 0.10
    box_r = 0.98
    
    box_dy = 0.05
    box_dx = box_dy * (h / w)
    box_y  = (box_t - box_b - 4*box_dy)/2
    box_x  = box_y * (h / w) / 2
    
    box_comp = dict(left=box_l,                  right=box_r,                  bottom=box_t-box_y,            top=box_t)
    box_rep1 = dict(left=box_l,                  right=box_l+2*box_x+1*box_dx, bottom=box_t-2*box_y-4*box_dy, top=box_t-box_y-3*box_dy)
    box_rep2 = dict(left=box_l+2*box_x+3*box_dx, right=box_l+4*box_x+4*box_dx, bottom=box_t-2*box_y-4*box_dy, top=box_t-box_y-3*box_dy)
    
    gs_comp = gridspec.GridSpec(1, 1, wspace=0, **box_comp)
    gs_rep1 = gridspec.GridSpec(2, 2, hspace=box_dy, wspace=box_dx, **box_rep1)
    gs_rep2 = gridspec.GridSpec(2, 2, hspace=box_dy, wspace=box_dx, **box_rep2)
    
    ax = plt.subplot(gs_comp[0, 0])
    
    # sublabels
    ldx = -0.05
    ldy = -0.01
    fig.text(box_comp['left']+ldx, box_comp['top']+ldy, s = 'a', **DEF_SUBLABEL, transform = fig.transFigure)
    fig.text(box_rep1['left']+ldx, box_rep1['top']+ldy, s = 'b', **DEF_SUBLABEL, transform = fig.transFigure)
    fig.text(box_rep2['left']+ldx, box_rep2['top']+ldy, s = 'c', **DEF_SUBLABEL, transform = fig.transFigure)

    # plots

    pop_list = pop_avg.items()
    pop_list = sorted(pop_list, key = lambda x: x[1], reverse = True)
    x, y = zip(*pop_list)
    ax.scatter(x, np.array(y)**2, color=C_POP, s=SMALLSIZEDOT*2)
    
    pref_list = pref_avg.items()
    x_, y_ = zip(*pref_list)
    ax.scatter(x_, np.array(y_)**2, color=C_PREF, s=SMALLSIZEDOT*2)
    # print([NAME2NAME[label] for label in x])
    # ax.xaxis.set_ticks([NAME2NAME[label] for label in x])
    ax.set_xticklabels([NAME2NAME[label] for label in x], rotation = 45, ha = 'right')
    
    # legend
    
#    ax.set_xlabel('Data set', fontsize = SIZELABEL)
#    ax.set_ylabel('Average Pearson correlation between\nreplicate inferred mutational effects', fontsize = SIZELABEL)
    ax.text(len(x)/2, -0.30, 'Data set', ha='center', va='center', **DEF_LABELPROPS)
    ax.set_ylabel('Average percent variantion explained\nbetween replicate inferred\nmutational effects, ' + r'$R^2$', fontsize = SIZELABEL)
    
#    ax.set_yticks([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0])
#    ax.set_ylim(0.25, 1.02)
    ax.set_yticks([0, 0.20, 0.40, 0.60, 0.80, 1.0])
#    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
#    ax.set_yminorticks([0.1, 0.30, 0.50, 0.70, 0.90])
    ax.set_ylim(0, 1)
    ax.set_xlim([-0.5, len(x)-0.5])
                    
    colors    = [C_POP, C_PREF]
    colors_lt = [C_POP, C_PREF]
    plotprops = DEF_ERRORPROPS.copy()
    plotprops['clip_on'] = False
    
    invt = ax.transData.inverted()
    xy1  = invt.transform((   0,  0))
    xy2  = invt.transform((7.50, 10))
    xy3  = invt.transform((3.00, 10))
    xy4  = invt.transform((5.25, 10))

    legend_dx1 = xy1[0]-xy2[0]
    legend_dx2 = xy1[0]-xy3[0]
    legend_dx  = xy1[0]-xy4[0]
    legend_dy  = xy1[1]-xy2[1]
    legend_x   = 0.2
    legend_y   = [0.12, 0.12 + 1*legend_dy]
    legend_t   = ['popDMS', 'Ratio/regression methods']
    
    for k in range(len(legend_y)):
        mp.error(ax=ax, x=[[legend_x+legend_dx]], y=[[legend_y[k]]], edgecolor=[colors[k]], facecolor=[colors_lt[k]], plotprops=plotprops)
        ax.text(legend_x, legend_y[k], legend_t[k], ha='left', va='center', **DEF_LABELPROPS)
    
    ax.spines['right'].set_visible(False)
    ax.spines[ 'top' ].set_visible(False)
    ax.tick_params(axis = 'both', which = 'major', labelsize = SIZELABEL)

    # Example inference with popDMS
    data_set = 'HIV Env BF520'
    n_reps   = 3

    df_temp  = pd.read_csv(POP_DIR + data_set + '.csv.gz')
    # df_temp  = df_temp[~(df_temp == 0).any(axis=1)]
    # df_temp  = df_temp[(df_temp['rep_1']!=0) & (df_temp['rep_2']!=0)]
    pop_list = []
    for rep in range(1, n_reps+1):
        pop_list.append(df_temp['rep_' + str(rep)].tolist())
        
    df_temp   = pd.read_csv(PREF_DIR + data_set + '.csv.gz')
    df_temp  = df_temp[~(df_temp == 0).any(axis=1)]
    pref_list = []
    for rep in range(1, n_reps+1):
        pref_list.append(df_temp['rep_' + str(rep)].tolist())
        # pref_list.append(list(temp_df.values.flatten()))

    scatterprops = dict(alpha=0.2, edgecolor='none', s=SMALLSIZEDOT)
    
    # FUTURE: MAKE A LOOP INSTEAD
    
    ticks = []
    lim   = [-1.1, 1.3]
    td    = (lim[1] - lim[0])*0.06
    
    n_digits = 2
    for i in range(n_reps):
        for j in range(i+1, n_reps):
            corr = round(st.pearsonr(pop_list[i], pop_list[j])[0], n_digits)
            ax2_sub = 0
            if i == 0 and j == 1:
                ax2_sub = plt.subplot(gs_rep1[0, 0])
                ax2_sub.set_ylabel('Replicate 2', fontsize = SIZELABEL)

            if i == 0 and j == 2:
                ax2_sub = plt.subplot(gs_rep1[1, 0])
                ax2_sub.set_ylabel('Replicate 3', fontsize = SIZELABEL)
                ax2_sub.set_xlabel('Replicate 1', fontsize = SIZELABEL)

            if i == 1 and j == 2:
                ax2_sub = plt.subplot(gs_rep1[1, 1])
                ax2_sub.set_xlabel('Replicate 2', fontsize = SIZELABEL)

            ax2_sub.scatter(pop_list[i], pop_list[j], **scatterprops)
            ax2_sub.text(lim[0]+td, lim[1]-td, 'R = %.2f' % corr, fontsize = SIZELABEL)

            ax2_sub.set_xticks(ticks)
            ax2_sub.set_yticks(ticks)
            ax2_sub.set_xlim(lim)
            ax2_sub.set_ylim(lim)

            ax2_sub.spines['right'].set_visible(False)
            ax2_sub.spines[ 'top' ].set_visible(False)
            
    ddx = 0.03
    ddy = ddx * (w / h)
    fig.text(box_l-ddx,                box_t-1.5*box_y-3.5*box_dy, s='popDMS', rotation=90, ha='center', va='center', **DEF_LABELPROPS, transform=fig.transFigure)
    fig.text(box_l+1*box_x+0.5*box_dx, box_b - ddy,                s='popDMS', rotation=0,  ha='center', va='center', **DEF_LABELPROPS, transform=fig.transFigure)

    # Scatter plot of sample experiment of Enrichment ratio
    
    ticks = []
    lim   = [-0.05, 1.05]
    td    = (lim[1] - lim[0])*0.05

    for i in range(3):
        for j in range(i + 1, 3):
            corr = round(st.pearsonr(pref_list[i], pref_list[j])[0], n_digits)
            ax3_sub = 0
            if i == 0 and j == 1:
                ax3_sub = plt.subplot(gs_rep2[0, 0])
                ax3_sub.set_ylabel('Replicate 2', fontsize = SIZELABEL)

            if i == 0 and j == 2:
                ax3_sub = plt.subplot(gs_rep2[1, 0])
                ax3_sub.set_ylabel('Replicate 3', fontsize = SIZELABEL)
                ax3_sub.set_xlabel('Replicate 1', fontsize = SIZELABEL)

            if i == 1 and j == 2:
                ax3_sub = plt.subplot(gs_rep2[1, 1])
                ax3_sub.set_xlabel('Replicate 2', fontsize = SIZELABEL)

            ax3_sub.scatter(pref_list[i], pref_list[j], c=C_PREF, **scatterprops)
            ax3_sub.text(lim[0]+td, lim[1]-td, 'R = %.2f' % corr, fontsize = SIZELABEL)

            ax3_sub.set_xticks(ticks)
            ax3_sub.set_yticks(ticks)
            ax3_sub.set_xlim(lim)
            ax3_sub.set_ylim(lim)

            ax3_sub.spines['right'].set_visible(False)
            ax3_sub.spines[ 'top' ].set_visible(False)
    
    fig.savefig(FIG_DIR + fig_title, dpi = 400, **FIGPROPS)
    plt.show()


def plot_selection(ax, df_pop, legend=True):
    """ Plot selection heatmap. """

    # process stored data

    sites = np.unique(df_pop[COL_SITE])
    
    df_WT   = df_pop[df_pop[COL_WT]==True]
    WT      = [df_WT[df_WT[COL_SITE]==s].iloc[0][COL_AA] for s in sites]
    s_WT    = [df_WT[df_WT[COL_SITE]==s].iloc[0][COL_S]  for s in sites]
    s_vec   = [[df_pop[(df_pop[COL_SITE]==sites[i]) & (df_pop[COL_AA]==aa)].iloc[0][COL_S] - s_WT[i] for aa in AA] for i in range(len(sites))]
    s_norm  = np.max(np.fabs(s_vec))

    # plot selection across the protein, normalizing WT residues to zero

    site_rec_props = dict(height=1, width=1, ec=None, lw=AXWIDTH/2, clip_on=False)
    prot_rec_props = dict(height=len(AA), width=len(sites), ec=BKCOLOR, fc='none', lw=AXWIDTH/2, clip_on=False)
    cBG             = '#F5F5F5'
    rec_patches     = []
    WT_dots_x       = []
    WT_dots_y       = []
    
    if legend:
        WT_dots_x.append(0.5)
        WT_dots_y.append(-3.5)
    
    for i in range(len(sites)):
        WT_dots_x.append(i + 0.5)
        WT_dots_y.append(len(AA)-AA.index(WT[i])+0.5)
        idxs     = sig_site_real==i+sub_i
        temp_s   = sig_s[idxs]
        temp_nuc = sig_nuc_idx[idxs]
        for j in range(len(AA)):
            
            # skip WT
            if AA[j]==WT[i]:
                continue

            # fill BG for unobserved
            c = cBG
            if s_vec[i][j]!=-s_WT[i]:
                t = s_vec[i][j] / s_norm
                if np.fabs(t)>1:
                    t /= np.fabs(t)
                if t>0:
                    c = hls_to_rgb(0.02, 0.53 * t + 1. * (1 - t), 0.83)
                else:
                    c = hls_to_rgb(0.58, 0.53 * np.fabs(t) + 1. * (1 - np.fabs(t)), 0.60)
            
            rec = matplotlib.patches.Rectangle(xy=(i, len(AA)-1-j), fc=c, **site_rec_props)
            rec_patches.append(rec)
    
    rec = matplotlib.patches.Rectangle(xy=(0, 0), **prot_rec_props)
    rec_patches.append(rec)
    
    # add patches and plot
    
    for patch in rec_patches:
        ax.add_artist(patch)

    pprops = { 'colors':    [BKCOLOR],
               'xlim':      [0, len(sites) + 1],
               'ylim':      [0, len(AA) + 0.5],
               'xticks':    [],
               'yticks':    [],
               'plotprops': dict(lw=0, s=0.2*SMALLSIZEDOT, marker='o', clip_on=False),
               'xlabel':    'Sites',
               'ylabel':    'Amino acids',
               'theme':     'open',
               'hide' :     ['top', 'bottom', 'left', 'right'] }

    mp.plot(type='scatter', ax=ax, x=[WT_dots_x], y=[WT_dots_y], **pprops)
    
    # legend
    
    rec_patches = []

    if legend:

        for i in range(-5, 5+1, 1):
            c = cBG
            t = i/5
            if t>0:
                c = hls_to_rgb(0.02, 0.53 * t + 1. * (1 - t), 0.83)
            else:
                c = hls_to_rgb(0.58, 0.53 * np.fabs(t) + 1. * (1 - np.fabs(t)), 0.60)
            rec = matplotlib.patches.Rectangle(xy=(len(sites) - 6.5 + i, -4), fc=c, **site_rec_props)
            rec_patches.append(rec)

        invt = ax.transData.inverted()
        xy1  = invt.transform((0,0))
        xy2  = invt.transform((0,9))
        legend_dy = (xy1[1]-xy2[1]) # multiply by 3 for slides/poster
        c   = cBG
        rec = matplotlib.patches.Rectangle(xy=(0, -4 + 4*legend_dy), fc=c, **site_rec_props) # paper
    #    rec = matplotlib.patches.Rectangle(xy=(0, -4 + 6*legend_dy), fc=c, **site_rec_props) # slides
        rec_patches.append(rec)

        for patch in rec_patches:
            ax.add_artist(patch)

        txtprops = dict(ha='center', va='center', color=BKCOLOR, family=FONTFAMILY, size=SIZELABEL)
#        for i in range(len(NUC)-1):
#            ax.text(-0.85, 3-i+0.5, NUC[i+1], clip_on=False, **txtprops)

        txtprops['ha'] = 'left'
        ax.text(1.3, -3.5, 'WT amino acid', clip_on=False, **txtprops)
        ax.text(1.3, -3.5 + 4*legend_dy, 'Not observed', clip_on=False, **txtprops) # paper
    #    ax.text(1.3, -3.5 + 6*legend_dy, 'Not observed', clip_on=False, **txtprops) # slides

        txtprops['ha'] = 'center'
        txtprops['va'] = 'top'
        for i in range(0, len(sites), 10):
            ax.text(1.5 + i, -0.5, sites[i], clip_on=False, **txtprops)

#        ax.text(-11, -4.5, -5, clip_on=False, **txtprops)
#        ax.text(- 6, -4.5,  0, clip_on=False, **txtprops)
#        ax.text(- 1, -4.5,  5, clip_on=False, **txtprops)
#        ax.text(- 5.5, -6.0, 'Inferred selection\ncoefficient, $\hat{s}$ (%)', clip_on=False, **txtprops)


######################
# FIGURE 1 FINITE SAMPLING SIMULATION
# FIGURE 1 ARGUMENTS

# FIGURE 1 MAIN PLOT FUNCTION
def FIG1_SIMULATION_FINITE_SAMPLING():

    FIG1_A_POS = {
        'x': 0.05,
        'y': 0.95
    }
    FIG1_B_POS = {
        'x': 0.05,
        'y': 0.48
    }
    FIG1_SIZE_X = SINGLE_COLUMN
    FIG1_SIZE_Y = SINGLE_COLUMN
    FIG1_SIMU_SCALE   = 'linear'
    FIG1_VAR          = 'R'
    FIG1_HSPACE       = 0.4
    FIG1_FINITE_NUM   = 10
    FIG1_MARKER_SIZE  = 5
    FIG1_GEN          = 10
    FIG1_ALPHA        = 0.3
    FIG1_BOX_ARG      = dict(left = 0.2, right = 0.95, bottom = 0.1, top = 0.95)
    FIG1_TRUE_COLOR   = COLOR_2
    FIG1_TRUE_ALPHA   = 1.0

    FIG1_TRAJECTORY_DIR = './output/simulation/WF_finite_sampling/'
    FIG1_INFERENCE_DIR = './output/simulation/WF_mutational_effects/'
    FIG1_WF_SIMU_FILE = './output/simulation/WF_simulation.csv'
    FIG1_WF_SELECTION = FIG1_INFERENCE_DIR + '/selection_coefficients/'
    FIG1_WF_LOGREG    = FIG1_INFERENCE_DIR+'/log_regression/'
    FIG1_WF_RATIO     = FIG1_INFERENCE_DIR+'/enrichment_ratio/'
    FIG1_WF_LOGRATIO  = FIG1_INFERENCE_DIR+'/enrichment_ratio_log/'
    # FIG1_FINITE_SIZE  = '_sampling-10000/'
    FIG1_NAME         = 'Fig1_simulation.pdf'

    fig = plt.figure(figsize = (FIG1_SIZE_X, FIG1_SIZE_Y))
    gs  = fig.add_gridspec(2, 1, hspace = FIG1_HSPACE, **FIG1_BOX_ARG)

    fig.text(**FIG1_A_POS, s = 'a', **DEF_SUBLABEL, transform = fig.transFigure)
    fig.text(**FIG1_B_POS, s = 'b', **DEF_SUBLABEL, transform = fig.transFigure)

    # Plot trajectories
    ax1 = fig.add_subplot(gs[:1, 0])
    ax1.set_xlabel('Generation', fontsize = SIZELABEL)
    ax1.set_ylabel('Variant frequency', fontsize = SIZELABEL)
    ax1.tick_params(axis = 'both', which = 'major', labelsize = SIZELABEL)
    ax1.set_yscale(FIG1_SIMU_SCALE)
    num_traj = 0

    for traj_num in range(FIG1_FINITE_NUM):
        df_trajectory = pd.read_csv(FIG1_TRAJECTORY_DIR + 'rep-%s'%traj_num + '.csv.zip', index_col=0)
        for i in [FIG1_VAR]:
            ax1.plot([i for i in range(FIG1_FINITE_NUM)], df_trajectory.loc[i][:FIG1_GEN], color = BKCOLOR, alpha = FIG1_ALPHA)
    df_trajectory = pd.read_csv(FIG1_WF_SIMU_FILE, index_col=0)
    col_list      = df_trajectory.T.columns.tolist()
    for i in [FIG1_VAR]:
        ax1.plot(df_trajectory.columns[:FIG1_GEN], df_trajectory.T[i][:FIG1_GEN], color = FIG1_TRUE_COLOR, alpha = FIG1_TRUE_ALPHA)

#    legend_fig1_a = [Line2D([0], [0], color = FIG1_FINITE_COLOR, lw = 2, label = 'Finitely sampled'),
#                     Line2D([0], [0], color = FIG1_TRUE_COLOR,   lw = 2, label = 'True evolution')]
#    ax1.legend(handles = legend_fig1_a, frameon = False, fontsize = SIZELABEL)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.set_yscale('log')
    ax1.set_xticks([0, 3, 6, 9])
    
    # legend
                    
    colors    = [FIG1_TRUE_COLOR, BKCOLOR]
    colors_lt = [FIG1_TRUE_COLOR, LCOLOR]
    plotprops = DEF_ERRORPROPS.copy()
    plotprops['clip_on'] = False
    
    invt = ax1.transData.inverted()
    xy1  = invt.transform((   0, 0))
    xy2  = invt.transform((7.50, 9))
    xy3  = invt.transform((3.00, 9))
    xy4  = invt.transform((5.25, 9))

    legend_dx1 = xy1[0]-xy2[0]
    legend_dx2 = xy1[0]-xy3[0]
    legend_dx  = xy1[0]-xy4[0]
    legend_dy  = -2.4e-4
    legend_x   =  0.015
    legend_y   = [1e-3, 1e-3 + legend_dy]
    legend_t   = ['True evolution', 'Finitely sampled']
    
    for k in range(len(legend_y)):
        mp.error(ax=ax1, x=[[legend_x+legend_dx]], y=[[legend_y[k]]], edgecolor=[colors[k]], facecolor=[colors_lt[k]], plotprops=plotprops)
        ax1.text(legend_x, legend_y[k], legend_t[k], ha='left', va='center', **DEF_LABELPROPS)
        
    # second panel

    ax2 = fig.add_subplot(gs[1:, 0])
    generation   = 10
    replicates   = 100
    sample_index = 10

    generations=[1, 4, 9, 19]
    finite_list = [50000]
    marker = ['o', 'v', '*']
    generation_ = [i + 1 for i in generations[: -1]]

    for finite_sampling in finite_list:
        temp = [[],[],[],[]]
        for generation in generation_:

            df_select               = pd.read_csv(FIG1_INFERENCE_DIR + '/selection_coefficients_gen-%s_'%generation+'sampling-%s'%finite_sampling+'.csv.zip', index_col=0)
            df_enrichment_regress   = pd.read_csv(FIG1_INFERENCE_DIR + '/log_regression_gen-%s_'%generation+'sampling-%s'%finite_sampling+'.csv.zip', index_col=0)
            df_enrichment_ratio     = pd.read_csv(FIG1_INFERENCE_DIR + '/enrichment_ratio_gen-%s_'%generation+'sampling-%s'%finite_sampling+'.csv.zip', index_col=0)
            df_enrichment_ratio_log = pd.read_csv(FIG1_INFERENCE_DIR + '/enrichment_ratio_log_gen-%s_'%generation+'sampling-%s'%finite_sampling+'.csv.zip', index_col=0)

            enrichment_ratio_corr       =     df_enrichment_ratio.T.corr(method = 'pearson')
            log_regression_corr         =   df_enrichment_regress.T.corr(method = 'pearson')
            selection_coefficients_corr =               df_select.T.corr(method = 'pearson')
            enrichment_ratio_log_corr   = df_enrichment_ratio_log.T.corr(method = 'pearson')
            
            factor = replicates * replicates - replicates
            enrichment_ratio_corr       = (enrichment_ratio_corr.sum().sum()       - replicates)/factor
            log_regression_corr         = (log_regression_corr.sum().sum()         - replicates)/factor
            selection_coefficients_corr = (selection_coefficients_corr.sum().sum() - replicates)/factor
            enrichment_ratio_log_corr   = (enrichment_ratio_log_corr.sum().sum()   - replicates)/factor
            
            
            temp[0].append(selection_coefficients_corr)
            temp[1].append(enrichment_ratio_corr)
            temp[2].append(log_regression_corr)
            temp[3].append(enrichment_ratio_log_corr)
            
        PALETTE = sns.hls_palette(3)

        ax2.plot(generation_, temp[0], c = C_POP,      marker = marker[finite_list.index(finite_sampling)], markersize = SMALLSIZEDOT, markeredgewidth = 0, alpha = 1)
        ax2.plot(generation_, temp[1], c = PALETTE[0], marker = marker[finite_list.index(finite_sampling)], markersize = SMALLSIZEDOT, markeredgewidth = 0, alpha = 1)
        ax2.plot(generation_, temp[2], c = PALETTE[1], marker = marker[finite_list.index(finite_sampling)], markersize = SMALLSIZEDOT, markeredgewidth = 0, alpha = 1)
        ax2.plot(generation_, temp[3], c = PALETTE[2], marker = marker[finite_list.index(finite_sampling)], markersize = SMALLSIZEDOT, markeredgewidth = 0, alpha = 1)
        ax2.set_xlabel('Generations used for inference',      fontsize = SIZELABEL)
        ax2.set_ylabel('Average Pearson correlation between\nreplicate inferred mutational effects', fontsize = SIZELABEL)
        ax2.set_xticks([2, 5, 10])
        ax2.set_xlim  ([   0, 11])
        ax2.set_ylim  ([-0.05, 1])
        
#    legend_fig_1b = [Line2D([0], [0], color = '#4F94CD', lw = 2, label = 'popDMS',           alpha = 0.6),
#                     Line2D([0], [0], color = 'red',     lw = 2, label = 'Enrichment ratio', alpha = 0.6),
#                     Line2D([0], [0], color = 'orange',  lw = 2, label = 'Log scaled ratio', alpha = 0.6),
#                     Line2D([0], [0], color = 'green',   lw = 2, label = 'Log regression',   alpha = 0.6)
#                    ]

    # legend
                    
    colors    = [C_POP, PALETTE[0], PALETTE[1], PALETTE[2]]
    colors_lt = [C_POP, PALETTE[0], PALETTE[1], PALETTE[2]]
    plotprops = DEF_ERRORPROPS.copy()
    plotprops['clip_on'] = False
    
    invt = ax2.transData.inverted()
    xy1  = invt.transform((   0,  0))
    xy2  = invt.transform((7.50, 10))
    xy3  = invt.transform((3.00, 10))
    xy4  = invt.transform((5.25, 10))

    legend_dx1 = xy1[0]-xy2[0]
    legend_dx2 = xy1[0]-xy3[0]
    legend_dx  = xy1[0]-xy4[0]
    legend_dy  = xy1[1]-xy2[1]
    legend_x   = 1.8
    legend_y   = [0.99, 0.99 + 1*legend_dy, 0.99 + 2*legend_dy, 0.99 + 3*legend_dy]
    legend_t   = ['popDMS', 'Enrichment ratio', 'Log ratio', 'Log regression']
    
    for k in range(len(legend_y)):
        mp.error(ax=ax2, x=[[legend_x+legend_dx]], y=[[legend_y[k]]], edgecolor=[colors[k]], facecolor=[colors_lt[k]], plotprops=plotprops)
        ax2.text(legend_x, legend_y[k], legend_t[k], ha='left', va='center', **DEF_LABELPROPS)

    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax2.set_xlim([1, 11])
    ax2.spines['right'].set_visible(False)
    ax2.spines[ 'top' ].set_visible(False)
    ax2.tick_params(axis = 'both', which = 'major', labelsize = SIZELABEL)
    #ax2.legend(handles = legend_fig_1b, frameon = False, fontsize = SIZELABEL, loc = 'upper left')
    fig.show()
    fig.savefig(FIG_DIR + FIG1_NAME, dpi = FIG_DPI, **FIGPROPS)


######################
# FIGURE 2 METHODS COMPARISON 
# FIGURE 2 ARGUMENTS

# FIGURE 2 MAIN PLOT FUNCTION
def FIG2_METHODS_COMPARISON():

    # variables
    FIG2_SIZE_X         = DOUBLE_COLUMN
    FIG2_SIZE_Y         = DOUBLE_COLUMN * 1.1 / GOLDR
    # C_POP    = C_POP
    FIG2_A_MPL_MARKER   = '.'
    C_PREF   = LCOLOR
    FIG2_A_PREF_MARKER  = '.'
    FIG2_A_MARKER_SIZE  = SMALLSIZEDOT*2
    FIG2_A_TAGBOX       = dict(boxstyle='round', facecolor = 'white')

    FIG2_A_INDEPENDENT_SITE_RESULT_DIR = {
                       'Flu_WSN':         ['WSN',                   '-2', 3],#
                       'Flu_A549':        ['A549',                  '-3', 2],#
                       'Flu_CCL141':      ['CCL141',                '-3', 3],#
                       'Flu_Aichi68C':    ['Aichi68C',              '-2', 2],#
                       'Flu_PR8':         ['PR8' ,                  '-3', 2],#
                       'Flu_MatrixM1':    ['Matrix_M1',             '-2', 3],#
                       'ZIKV':            ['ZIKV',                  '-2', 3],
                       'Perth2009':       ['Perth2009',             '-3', 4],
                       'Flu_MS':          ['MS',                    '-3', 2],#
                       'Flu_MxA':         ['MxA',                   '-3', 2],#
                       'Flu_MxAneg':      ['MxAneg',                '-3', 2],#
                       'HIV_BG505':       ['HIV Env BG505' ,        '-3', 3],#
                       'HIV_BF520':       ['HIV Env BF520' ,        '-3', 3],#
                       'HIV_CD4_human':   ['HIV BF520 human host',  '-3', 2],#
                       'HIV_CD4_rhesus':  ['HIV BF520 rhesus host', '-2', 2],#
                       'HIV_bnAbs_FP16':  ['HIV bnAbs FP16',        '-2', 2],
                       'HIV_bnAbs_FP20':  ['HIV bnAbs FP20',        '-2', 2],
                       'HIV_bnAbs_VRC34': ['HIV bnAbs VRC34',       '-2', 2],
                       }
    
    FIG2_A_FULL_LENGTH_RESULT_DIR = {
                       'HDR_Y2H_1':      ['Y2H_1',         '-1', 3],
                       'HDR_Y2H_2':      ['Y2H_2',         '-1', 3],
                       'HDR_E3':         ['E3',            '-1', 6],
                       'WWdomain_YAP1':  ['YAP1',          '-3', 2],
                       'Ubiq_Ube4b':     ['Ube4b',         '-4', 2],
                       'HDR_DBR1':       ['DBR1',          '1',  2],
                       'Thrombo_TpoR_1': ['TpoR',          '-1', 6],
                       'Thrombo_TpoR_2': ['TpoR_S505N',    '-1', 6],
                       }

    FIG2_B_REPLICATE_NUM = 3
#     FIG2_B_SAMPLE_DIR    = './output/virus_protein/HIVEnv/BF520/selection_coefficients/'
    FIG2_B_SAMPLE_DIR   = './output/selection_coefficients/'
    FIG2_B_SCATTER_SIZE  = SMALLSIZEDOT
    FIG2_B_SCATTER_ALPHA = 0.2
    FIG2_B_CORR_DIGIT    = 2

    FIG2_C_REPLICATE_NUM = 3
#     FIG2_C_SAMPLE_DIR    = './data/virus_protein/HIVEnv/BF520/pref/'
    FIG2_C_SAMPLE_DIR   = './data/prefs/'
    FIG2_C_SCATTER_SIZE  = SMALLSIZEDOT
    FIG2_C_SCATTER_ALPHA = 0.2
    FIG2_C_CORR_DIGIT    = 2

    FIG2_NAME            = 'Fig2_comparison.pdf'

    FIG2_A_PREF_AVG = {}
    FIG2_A_MPL_AVG  = {}
    for target_protein, info in FIG2_A_INDEPENDENT_SITE_RESULT_DIR.items():
        path = POP_DIR +  info[0] + '.csv.gz'
        df_temp = pd.read_csv(path)
        df_temp = df_temp[(df_temp['rep_1'] != 0) & (df_temp['rep_2'] != 0)]
        df_corr = df_temp[df_temp.columns[2:]]
        correlation_average = (df_corr.corr().sum().sum() - df_corr.shape[1])/(df_corr.shape[1]**2 - df_corr.shape[1])
        FIG2_A_MPL_AVG[target_protein] = correlation_average
                
    for protein, info_list in FIG2_A_INDEPENDENT_SITE_RESULT_DIR.items():
        SELECTION_LIST = []
        ENRICH_LIST    = []
        replicate      = info_list[2]
        if 'HIV_bnAbs' in protein or 'Flu_MS' in protein or 'Flu_Mx' in protein:
            FILE_PATH = PREF_DIR + info_list[0] + '.csv.gz'
            temp_df   = pd.read_csv(FILE_PATH)
            correlation_average      = (temp_df.corr().sum().sum() - temp_df.shape[1])/(temp_df.shape[1]**2 - temp_df.shape[1])
            FIG2_A_PREF_AVG[protein] = correlation_average
        else:
            for rep in range(replicate):
                FILE_PATH = PREF_DIR + info_list[0] + '.csv.gz'
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


    PERFORMANCE_FIG_SIZE = (FIG2_SIZE_X, FIG2_SIZE_Y)
    fig = plt.figure(figsize = PERFORMANCE_FIG_SIZE)
    
    box_t = 0.98
    box_b = 0.07
    box_l = 0.10
    box_r = 0.98
    
    box_dy = 0.05
    box_dx = box_dy * (FIG2_SIZE_Y / FIG2_SIZE_X)
    box_y  = (box_t - box_b - 4*box_dy)/2
    box_x  = box_y * (FIG2_SIZE_Y / FIG2_SIZE_X) / 2
    
    box_comp = dict(left=box_l,                  right=box_r,                  bottom=box_t-box_y,            top=box_t)
    box_rep1 = dict(left=box_l,                  right=box_l+2*box_x+1*box_dx, bottom=box_t-2*box_y-4*box_dy, top=box_t-box_y-3*box_dy)
    box_rep2 = dict(left=box_l+2*box_x+3*box_dx, right=box_l+4*box_x+4*box_dx, bottom=box_t-2*box_y-4*box_dy, top=box_t-box_y-3*box_dy)
    
    gs_comp = gridspec.GridSpec(1, 1, wspace=0, **box_comp)
    gs_rep1 = gridspec.GridSpec(2, 2, hspace=box_dy, wspace=box_dx, **box_rep1)
    gs_rep2 = gridspec.GridSpec(2, 2, hspace=box_dy, wspace=box_dx, **box_rep2)
    
    ax = plt.subplot(gs_comp[0, 0])
    
    # sublabels
    ldx = -0.05
    ldy = -0.01
    fig.text(box_comp['left']+ldx, box_comp['top']+ldy, s = 'a', **DEF_SUBLABEL, transform = fig.transFigure)
    fig.text(box_rep1['left']+ldx, box_rep1['top']+ldy, s = 'b', **DEF_SUBLABEL, transform = fig.transFigure)
    fig.text(box_rep2['left']+ldx, box_rep2['top']+ldy, s = 'c', **DEF_SUBLABEL, transform = fig.transFigure)
    
#    ax  = fig.add_subplot(gs[:2,:4])
#    ax2 = fig.add_subplot(gs[3:,:2])
#    ax3 = fig.add_subplot(gs[3:,2:])
#    FIG2_B_INNER = gridspec.GridSpecFromSubplotSpec(2, 2,
#                                                    hspace = 0.2,
#                                                    wspace = 0,
#                                                    subplot_spec  = ax2,
#                                                    height_ratios = [1, 1], width_ratios = [1, 1])
#
#    FIG2_C_INNER = gridspec.GridSpecFromSubplotSpec(2, 2,
#                                                    hspace = 0.2,
#                                                    wspace = 0,
#                                                    subplot_spec  = ax3,
#                                                    height_ratios = [1, 1], width_ratios = [1, 1])

    # human
    #FIG2_A_PREF_AVG = {}
    #FIG2_A_MPL_AVG = {}
    #FIG2_A_MPL_AVG[' '] = 100
#     for target_protein, info in FIG2_A_HUMAN_RESULT_DIR.items():
#         path = FIG2_HUMAN_DIR+info[0] + 'selection_coefficients/'
#         for file in os.listdir(path):
#             if file.endswith('.csv.gz'):
#                 df_temp = pd.read_csv(path + file)
#                 df_temp = df_temp[(df_temp['rep_1'] != 0)&(df_temp['rep_2'] != 0)]
#                 df_corr = df_temp[df_temp.columns[2:]]
#                 correlation_average = (df_corr.corr().sum().sum() - df_corr.shape[1])/(df_corr.shape[1]**2 - df_corr.shape[1])
#                 FIG2_A_MPL_AVG[target_protein] = correlation_average
    for target_protein, info in FIG2_A_FULL_LENGTH_RESULT_DIR.items():
        path = POP_DIR +  info[0] + '.csv.gz'
        df_temp = pd.read_csv(path)
        df_temp = df_temp[(df_temp['rep_1'] != 0) & (df_temp['rep_2'] != 0)]
        df_corr = df_temp[df_temp.columns[2:]]
        correlation_average = (df_corr.corr().sum().sum() - df_corr.shape[1])/(df_corr.shape[1]**2 - df_corr.shape[1])
        FIG2_A_MPL_AVG[target_protein] = correlation_average

    for protein, info_list in FIG2_A_FULL_LENGTH_RESULT_DIR.items():
#         SELECTION_LIST = []
#         ENRICH_LIST    = []
#         replicate      = info_list[1]
        
        FILE_PATH = PREF_DIR + info_list[0] + '_prefs.csv.gz'
        temp_df = pd.read_csv(FILE_PATH, index_col = 0)
        if 'hgvs_pro' in temp_df.columns:
            temp_df = temp_df[~temp_df['hgvs_pro'].str.contains('\[')]
        df_corr = temp_df[temp_df.columns.tolist()[-info_list[2]:]]
        df_corr = df_corr.dropna()
        correlation_average = (df_corr.corr().sum().sum() - df_corr.shape[1])/(df_corr.shape[1]**2 - df_corr.shape[1])
        FIG2_A_PREF_AVG[protein] = correlation_average

    MPL_LIST = FIG2_A_MPL_AVG.items()
    MPL_LIST = sorted(MPL_LIST, key = lambda x: x[1], reverse = True)
    x, y = zip(*MPL_LIST)
#    ax.scatter(x, y, color = C_POP, s = FIG2_A_MARKER_SIZE)
    ax.scatter(x, np.array(y)**2, color = C_POP, s = FIG2_A_MARKER_SIZE)
    
    ENRICH_LIST = FIG2_A_PREF_AVG.items()
    x_, y_ = zip(*ENRICH_LIST)
#    ax.scatter(x_, y_, color = C_PREF, s = FIG2_A_MARKER_SIZE)
    ax.scatter(x_, np.array(y_)**2, color = C_PREF, s = FIG2_A_MARKER_SIZE)
    
#    for i in range(len(x)):
#        print('%s\t%s' % (x[i], y[i]))
#
#    print('')
#
#    for i in range(len(x_)):
#        print('%s\t%s' % (x_[i], y_[i]))
    
    #labels = [label.replace('_', ' ') for label in x]
    ax.set_xticklabels([NAME2NAME[label] for label in x], rotation = 45, ha = 'right')
    
    # legend
    
#    ax.set_xlabel('Data set', fontsize = SIZELABEL)
#    ax.set_ylabel('Average Pearson correlation between\nreplicate inferred mutational effects', fontsize = SIZELABEL)
    ax.text(len(x)/2, -0.30, 'Data set', ha='center', va='center', **DEF_LABELPROPS)
    ax.set_ylabel('Average percent variantion explained\nbetween replicate inferred\nmutational effects, ' + r'$R^2$', fontsize = SIZELABEL)
    
#    ax.set_yticks([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0])
#    ax.set_ylim(0.25, 1.02)
    ax.set_yticks([0, 0.20, 0.40, 0.60, 0.80, 1.0])
#    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
#    ax.set_yminorticks([0.1, 0.30, 0.50, 0.70, 0.90])
    ax.set_ylim(0, 1)
    ax.set_xlim([-0.5, len(x)-0.5])
                    
    colors    = [C_POP, C_PREF]
    colors_lt = [C_POP, C_PREF]
    plotprops = DEF_ERRORPROPS.copy()
    plotprops['clip_on'] = False
    
    invt = ax.transData.inverted()
    xy1  = invt.transform((   0,  0))
    xy2  = invt.transform((7.50, 10))
    xy3  = invt.transform((3.00, 10))
    xy4  = invt.transform((5.25, 10))

    legend_dx1 = xy1[0]-xy2[0]
    legend_dx2 = xy1[0]-xy3[0]
    legend_dx  = xy1[0]-xy4[0]
    legend_dy  = xy1[1]-xy2[1]
    legend_x   = 0.2
    legend_y   = [0.12, 0.12 + 1*legend_dy]
    legend_t   = ['popDMS', 'Ratio/regression methods']
    
    for k in range(len(legend_y)):
        mp.error(ax=ax, x=[[legend_x+legend_dx]], y=[[legend_y[k]]], edgecolor=[colors[k]], facecolor=[colors_lt[k]], plotprops=plotprops)
        ax.text(legend_x, legend_y[k], legend_t[k], ha='left', va='center', **DEF_LABELPROPS)
    
    ax.spines['right'].set_visible(False)
    ax.spines[ 'top' ].set_visible(False)
    ax.tick_params(axis = 'both', which = 'major', labelsize = SIZELABEL)

    # Scatter plot of sample experiment of MPL
    SELECTION_LIST = []
#     for file in os.listdir(FIG2_B_SAMPLE_DIR):
#         if file.endswith('.csv.gz'):
#             df_temp = pd.read_csv(FIG2_B_SAMPLE_DIR+file)
#             df_temp = df_temp[(df_temp['rep_1'] != 0)&(df_temp['rep_2'] != 0)]
#             for rep in range(1, FIG2_B_REPLICATE_NUM + 1):
#                 SELECTION_LIST.append(df_temp['rep_' + str(rep)].tolist())

    df_temp = pd.read_csv(FIG2_B_SAMPLE_DIR+'HIV Env BF520.csv.gz')
    df_temp = df_temp[(df_temp['rep_1'] != 0)&(df_temp['rep_2'] != 0)]
    for rep in range(1, FIG2_B_REPLICATE_NUM + 1):
        SELECTION_LIST.append(df_temp['rep_' + str(rep)].tolist())

    SCATTER_DOT = {
        'alpha': 0.2,
        'edgecolor': 'none',
        's': FIG2_B_SCATTER_SIZE
    }
    
    ticks = []
    lim = [-1.1, 1.3]
    td  = (lim[1] - lim[0])*0.06
    
    for i in range(3):
        for j in range(i + 1, 3):
            CORR = round(st.pearsonr(SELECTION_LIST[i], SELECTION_LIST[j])[0], FIG2_B_CORR_DIGIT)
            ax2_sub = 0
            if i == 0 and j == 1:
                ax2_sub = plt.subplot(gs_rep1[0, 0])
                ax2_sub.set_ylabel('Replicate 2', fontsize = SIZELABEL)

            if i == 0 and j == 2:
                ax2_sub = plt.subplot(gs_rep1[1, 0])
                ax2_sub.set_ylabel('Replicate 3', fontsize = SIZELABEL)
                ax2_sub.set_xlabel('Replicate 1', fontsize = SIZELABEL)

            if i == 1 and j == 2:
                ax2_sub = plt.subplot(gs_rep1[1, 1])
                ax2_sub.set_xlabel('Replicate 2', fontsize = SIZELABEL)

            ax2_sub.scatter(SELECTION_LIST[i], SELECTION_LIST[j], **SCATTER_DOT)
            ax2_sub.text(lim[0]+td, lim[1]-td, 'R = %.2f' %CORR, fontsize = SIZELABEL)

            ax2_sub.set_xticks(ticks)
            ax2_sub.set_yticks(ticks)
            ax2_sub.set_xlim(lim)
            ax2_sub.set_ylim(lim)

            ax2_sub.spines['right'].set_visible(False)
            ax2_sub.spines[ 'top' ].set_visible(False)
            
    ddx = 0.03
    ddy = ddx * (FIG2_SIZE_X / FIG2_SIZE_Y)
    fig.text(box_l-ddx,                box_t-1.5*box_y-3.5*box_dy, s='popDMS', rotation=90, ha='center', va='center', **DEF_LABELPROPS, transform=fig.transFigure)
    fig.text(box_l+1*box_x+0.5*box_dx, box_b - ddy,                s='popDMS', rotation=0,  ha='center', va='center', **DEF_LABELPROPS, transform=fig.transFigure)

    # Scatter plot of sample experiment of Enrichment ratio
    ENRICHMENT_LIST = []
    for rep in range(FIG2_C_REPLICATE_NUM):
        FILE_PATH = FIG2_C_SAMPLE_DIR + 'HIV Env BF520-' + str(rep+1) + '_prefs.csv'
        temp_df   = pd.read_csv(FILE_PATH, index_col = 0)
        ENRICHMENT_LIST.append(list(temp_df.values.flatten()))
#     for file in os.listdir(FIG2_C_SAMPLE_DIR):
#         if file.endswith('.csv'):
#             df_temp = pd.reCd_csv(FIG2_C_SAMPLE_DIR+file)
#             df_temp = df_temp.drop('site', axis = 1)
#             ENRICHMENT_LIST.append(df_temp.to_numpy().flatten())
#     for file in os.listdir(FIG2_C_SAMPLE_DIR):
#         if file.endswith('.csv'):
#             df_temp = pd.read_csv(FIG2_C_SAMPLE_DIR+file)
#             df_temp = df_temp.drop('site', axis = 1)
#             ENRICHMENT_LIST.append(df_temp.to_numpy().flatten())

    SCATTER_DOT_ENRICH = {
        'alpha': 0.2,
        'edgecolor': 'none',
        's': FIG2_B_SCATTER_SIZE,
    }
    
    ticks = []
    lim = [-0.05, 1.05]
    td  = (lim[1] - lim[0])*0.05

    for i in range(3):
        for j in range(i + 1, 3):
            CORR = round(st.pearsonr(ENRICHMENT_LIST[i], ENRICHMENT_LIST[j])[0], FIG2_B_CORR_DIGIT)
            ax3_sub = 0
            if i == 0 and j == 1:
                ax3_sub = plt.subplot(gs_rep2[0, 0])
                ax3_sub.set_ylabel('Replicate 2', fontsize = SIZELABEL)

            if i == 0 and j == 2:
                ax3_sub = plt.subplot(gs_rep2[1, 0])
                ax3_sub.set_ylabel('Replicate 3', fontsize = SIZELABEL)
                ax3_sub.set_xlabel('Replicate 1', fontsize = SIZELABEL)

            if i == 1 and j == 2:
                ax3_sub = plt.subplot(gs_rep2[1, 1])
                ax3_sub.set_xlabel('Replicate 2', fontsize = SIZELABEL)

            ax3_sub.scatter(ENRICHMENT_LIST[i], ENRICHMENT_LIST[j], c=LCOLOR, **SCATTER_DOT)
            ax3_sub.text(lim[0]+td, lim[1]-td, 'R = %.2f' %CORR, fontsize = SIZELABEL)

            ax3_sub.set_xticks(ticks)
            ax3_sub.set_yticks(ticks)
            ax3_sub.set_xlim(lim)
            ax3_sub.set_ylim(lim)

            ax3_sub.spines['right'].set_visible(False)
            ax3_sub.spines[ 'top' ].set_visible(False)

#    for i in range(3):
#        for j in range(i + 1, 3):
#            CORR = round(st.pearsonr(ENRICHMENT_LIST[i], ENRICHMENT_LIST[j])[0], FIG2_B_CORR_DIGIT)
#            if i == 0 and j == 1:
#                ax3_sub = plt.Subplot(fig, FIG2_C_INNER[0, 0])
#                ax3_sub.scatter(ENRICHMENT_LIST[i], ENRICHMENT_LIST[j], **SCATTER_DOT_ENRICH)
#                ax3_sub.set_ylabel('Replicate 2', fontsize = SIZELABEL)
#                ax3_sub.set_yticks([0, 0.5, 1])
#                ax3_sub.set_xticks([0, 0.5, 1])
#                ax3_sub.set_xticklabels([])
#
#
#            if i == 0 and j == 2:
#                ax3_sub = plt.Subplot(fig, FIG2_C_INNER[1, 0])
#                ax3_sub.scatter(ENRICHMENT_LIST[i], ENRICHMENT_LIST[j], **SCATTER_DOT_ENRICH)
#                ax3_sub.set_ylabel('Replicate 3', fontsize = SIZELABEL)
#                ax3_sub.set_xlabel('Replicate 1', fontsize = SIZELABEL)
#                ax3_sub.set_yticks([0, 0.5, 1])
#                ax3_sub.set_xticks([0, 0.5, 1])
#
#            if i == 1 and j == 2:
#                ax3_sub = plt.Subplot(fig, FIG2_C_INNER[1, 1])
#                ax3_sub.scatter(ENRICHMENT_LIST[i], ENRICHMENT_LIST[j], **SCATTER_DOT_ENRICH)
#                ax3_sub.set_xlabel('Replicate 2', fontsize = SIZELABEL)
#                ax3_sub.set_yticks([0, 0.5, 1])
#                ax3_sub.set_yticklabels([])
#                ax3_sub.set_xticks([0, 0.5, 1])
#
#            ax3_sub.text(0.0, 1.1, 'R = %.2f' %CORR, fontsize = SIZELABEL)
#
#            ax3_sub.set_xlim([-0.1, 1.3])
#            ax3_sub.set_ylim([-0.1, 1.3])
#            fig.add_subplot(ax3_sub)
#            ax3_sub.spines['right'].set_visible(False)
#            ax3_sub.spines[ 'top' ].set_visible(False)
#            ax3_sub.tick_params(axis = 'both', which = 'major', labelsize = SIZELABEL)
#            ax3_sub.set_aspect('equal')
#    ax3.set_xlabel('Enrichment ratio', fontsize = SIZELABEL, labelpad = 25)
#    ax3.set_ylabel('Enrichment ratio', fontsize = SIZELABEL, labelpad = 25)
    
    fig.savefig(FIG_DIR + FIG2_NAME, dpi = 400, **FIGPROPS)
    plt.show()


#################
# FIGURE 3 VISUALIZATION 
# FIGURE 3 ARGUMENTS

def FIG3_VISUALIZATION(exp_scale = 10, sites_per_line = 35):

    matplotlib.rcParams.update({'font.size': SIZELABEL})
    OFFSET_LETTER  = 0
    SELECTION_FILE = './output/selection_coefficients/YAP1_fig3.csv.gz'
    EPISTASIS_FILE = './output/epistasis/YAP1_100.txt'
    INDEX_FILE     = './output/epistasis/index_matrix.csv'
    SEQUENCE       = "DVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPR"
    PAPER_FIGURE_SIZE_X = 18
    PAPER_FIGURE_SIZE_Y = 14
    EXAMPLE_FIG_SIZE    = (PAPER_FIGURE_SIZE_X, PAPER_FIGURE_SIZE_Y)

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
    
    fig = plt.figure(figsize = EXAMPLE_FIG_SIZE)
    text_in_figure = {
        'fontsize': SIZELABEL,
        'bbox'    : {'facecolor': 'none', 'edgecolor': 'none', 'boxstyle': 'round'},
        'ha'      : 'center',
        'va'      : 'center'
    }
    fig.text(0.02, 0.94, s = 'a', **DEF_SUBLABEL, transform = fig.transFigure)
    fig.text(0.47, 0.94, s = 'b', **DEF_SUBLABEL, transform = fig.transFigure)
    fig.text(0.02, 0.52, s = 'c', **DEF_SUBLABEL, transform = fig.transFigure)
    fig.text(0.47, 0.52, s = 'd', **DEF_SUBLABEL, transform = fig.transFigure)
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
            logo.ax.set_ylabel("Normalized \nselection coefficients", fontsize = SIZELABEL)
            logo.ax.set_xlabel("Site", fontsize = SIZELABEL)
            
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
            clb.ax.set_xlabel('Selection\ncoefficient', fontsize = SIZELABEL)
            index_of_element_to_outline = []

            for j in range(len(site_sub)):
                index_of_element_to_outline.append([j - 0.5 - left_n, AA.index(SEQUENCE[site_list.index(site_sub[j])]) - 0.5])
           
            for outliner in index_of_element_to_outline:
                ax2.scatter(outliner[0] + 0.5, outliner[1] + 0.5, c = 'black', s = 2)
                
        ax2.set_xlabel("Site", fontsize = SIZELABEL)
        ax2.set_ylabel("Amino acid", fontsize = SIZELABEL)
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
        ax2.legend(handles = legend_elements, loc = [0.4, 1.05], ncol = 2, frameon = False, fontsize = SIZELABEL, handletextpad = 0.1)
        ax2.set_position([0.52, 0.6, 0.37, 0.37])

        # Comparison Logoplot
        inner = gridspec.GridSpecFromSubplotSpec(3, 1,subplot_spec = ax3, wspace = 0.1, hspace = 0.5)
        df_enrich   = pd.read_csv('./data/prefs/YAP1_prefs.csv.gz')
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
                # df_all = df_all.concat(df_zero)
                df_all = pd.concat([df_all, df_zero])
            m += 1
            # df_all    = df_all.concat(df_zero)
            df_all = pd.concat([df_all, df_zero])
            df_temp   = data1[data1['site'] == str(i) + '_MPL'].copy()
            temp_pure = df_temp[df_temp.columns[1:]]
            
            temp_pure = np.exp(MPL_scale * temp_pure)
            temp_pure.replace(1, 0, inplace = True)
            temp_pure = temp_pure.div(temp_pure.sum(axis = 1), axis = 0)
            # df_all = df_all.concat(temp_pure)
            df_all = pd.concat([df_all, temp_pure])
            
            #PREF
            df_temp   = data2[data2['site'] == str(i) + '_Pref'].copy()
            temp_pure = df_temp[df_temp.columns[1:]]
            temp_pure = np.exp(PREF_scale * temp_pure)
            temp_pure.replace(1, 0, inplace = True)
            temp_pure = temp_pure.div(temp_pure.sum(axis = 1), axis = 0)
            # df_all = df_all.concat(temp_pure)
            df_all = pd.concat([df_all, temp_pure])
        
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
                logo.ax.text(2.5, 1.1, 'Site 18(E)', **text_in_figure)
                logo.ax.text(5.5, 1.1, 'Site 24(S)', **text_in_figure)
                logo.ax.text(8.5, 1.1, 'Site 35(Q)', **text_in_figure)
                logo.ax.text(-.5, .5, 'Similar\ninference\n(Wild Type)', **text_in_figure)
                
            if i == 1:
                logo.ax.text(2.5, 1.1, 'Site 12(P)', **text_in_figure)
                logo.ax.text(5.5, 1.1, 'Site 21(K)', **text_in_figure)
                logo.ax.text(8.5, 1.1, 'Site 36(T)', **text_in_figure)
                logo.ax.set_ylabel("Normalized measurements", fontsize = SIZELABEL, labelpad=10)
                logo.ax.text(-.5, .5, 'Similar\ninference\n(Tolerance)', 
                             fontsize = SIZELABEL,
                             bbox=dict(facecolor = 'none', 
                                       edgecolor = 'none', 
                                       boxstyle  = 'round'), 
                             ha = 'center', va = 'center')

            else:
                logo.ax.set_ylabel(" ", fontsize = SIZELABEL, labelpad = 3)
            if i == 2:
                logo.ax.text(2.5, 1.1, 'Site 37(T)', **text_in_figure)
                logo.ax.text(5.5, 1.1, 'Site 38(T)', **text_in_figure)
                logo.ax.text(8.5, 1.1, 'Site 42(P)', **text_in_figure)
                logo.ax.text(-.5, .5, 'Different\ninference', 
                             fontsize = SIZELABEL,
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
        clb.ax.set_xlabel(r'$\sum$|epistasis|', fontsize = SIZELABEL)
        
    plt.savefig(FIG_DIR + 'Fig3_realdata.pdf', dpi=200)
    plt.show()


def FIG4_VISUALIZATION():

    FIG4_SIZE_X        = 20
    FIG4_SIZE_Y        = 15
    FIG4_C_LEGEND_XY   = [0.4, 0.6]
    FIG4_BOX_ARG       = dict(left = 0.12, right = 0.95, bottom = 0.3, top = 0.85)
    EPISTASIS_FIG_SIZE = (FIG4_SIZE_X, FIG4_SIZE_Y)


    matplotlib.rc_file_defaults()
    matplotlib.rc('text', usetex=False)
    matplotlib.rc('font', **DEF_LABELPROPS)

    fig = plt.figure(figsize = EPISTASIS_FIG_SIZE)
    gs  = fig.add_gridspec(11, 12, wspace = 1.3, **FIG4_BOX_ARG)
    gs.update(wspace=0.5, hspace=0.5)
    ax  = fig.add_subplot(gs[:11,:7])
    ax2 = fig.add_subplot(gs[:6,8:])
    ax3 = fig.add_subplot(gs[7:,8:])

    df_merged = pd.read_csv('./data/epistasis/merged_table.csv')
    df_merged.loc[(df_merged['site_1_y']==df_merged['site_2_y'])&(df_merged['AA_1_y']==df_merged['AA_2_y'])]
    df_new = df_merged.copy()[['site_1_y','site_2_y','epistasis_paper','epistasis_MPL_cons_1']].dropna()
    df_new['epistasis_paper'] = np.abs(df_new['epistasis_paper'])
    df_new['epistasis_MPL_cons_1'] = np.abs(df_new['epistasis_MPL_cons_1'])
    df_new=df_new.groupby(['site_1_y', 'site_2_y'])['epistasis_paper', 'epistasis_MPL_cons_1'].agg('sum').reset_index()
    df_new['epistasis_paper'] = (df_new['epistasis_paper'] - df_new['epistasis_paper'].min())/(df_new['epistasis_paper'].max() - df_new['epistasis_paper'].min())
    df_new['epistasis_MPL_cons_1'] = (df_new['epistasis_MPL_cons_1'] - df_new['epistasis_MPL_cons_1'].min())/(df_new['epistasis_MPL_cons_1'].max() - df_new['epistasis_MPL_cons_1'].min())

    x_list = [i + 2 for i in range(34)]
    df_heatmap = pd.DataFrame(columns=x_list)
    for i in (x_list):
        df_heatmap[i]=np.zeros(36)
    for index, row in df_new.iterrows():
        df_heatmap.loc[row['site_1_y']+1,row['site_2_y']+1]=row['epistasis_paper']
        df_heatmap.loc[row['site_2_y']+1,row['site_1_y']+1]=row['epistasis_MPL_cons_1']
    df_heatmap=df_heatmap.iloc[2:]
    sns.heatmap(df_heatmap,cmap='Blues',cbar_kws={'label': r'$\sum$|epistasis|'}, ax=ax)
    ax.set_xlabel('site',fontsize = SIZELABEL)
    ax.set_ylabel('site',fontsize = SIZELABEL)
    ax.set_xlabel('Site', labelpad = 7)
    ax.set_ylabel('Site', labelpad = 7)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_xticks([i for i in range(1,len(SEQUENCE)) if (i)%5 == 0])
    ax.set_xticklabels([i + 10 for i in range(1,len(SEQUENCE)) if (i)%5 == 0], fontsize = SIZELABEL)
    ax.set_yticks([i for i in range(1,len(SEQUENCE)) if (i)%5 == 0])
    ax.set_yticklabels([i + 10 for i in range(1,len(SEQUENCE)) if (i)%5 == 0], fontsize = SIZELABEL)
    ax.tick_params(axis = u'both', which = u'both', length = 0)
    ax.invert_yaxis()


    df_epistasis = df_merged.dropna().copy()
    df_epistasis['epistasis_MPL_absolute']   = np.abs(df_epistasis['epistasis_MPL_cons_1'])
    df_epistasis['epistasis_paper_absolute'] = np.abs(df_epistasis['epistasis_paper'])

    df_epistasis = df_epistasis[['site_1_y', 'site_2_y', 'epistasis_MPL_cons_1', 'epistasis_MPL_absolute', 'epistasis_paper','epistasis_paper_absolute', 'distance']].sort_values('epistasis_MPL_absolute')
    df_unique = df_epistasis[['site_1_y','site_2_y','distance','epistasis_MPL_absolute','epistasis_paper_absolute']]
    df_MPL = df_unique.groupby(['site_1_y','site_2_y','distance'])['epistasis_MPL_absolute', 'epistasis_paper_absolute'].agg('sum').reset_index().sort_values('epistasis_MPL_absolute')
    df_func = df_unique.groupby(['site_1_y','site_2_y','distance'])['epistasis_MPL_absolute', 'epistasis_paper_absolute'].agg('sum').reset_index().sort_values('epistasis_paper_absolute')
    
    df_epistasis.plot.scatter(x = 'epistasis_paper', y = 'epistasis_MPL_cons_1', alpha = 0.1, ax = ax2, s = 5)
    pr = st.pearsonr(df_epistasis.dropna()['epistasis_paper'], df_epistasis.dropna()['epistasis_MPL_cons_1'])[0]
    sr = st.spearmanr(df_epistasis.dropna()['epistasis_paper'], df_epistasis.dropna()['epistasis_MPL_cons_1'])[0]
    ax2.set_ylabel('popDMS epistasis',    fontsize = SIZELABEL)
    ax2.set_xlabel('reference epistasis', fontsize = SIZELABEL)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.text(-1.5, 1.5, s = 'Pearsonr = %.2f'%pr, fontsize = SIZELABEL)
    ax2.text(-1.5, 1.3, s = 'Spearmanr = %.2f'%sr, fontsize = SIZELABEL)
    ax2.set_yticks([0,0.5,1,1.5])
    fig.text(0.08, 0.85, s = 'a', **DEF_SUBLABEL, transform = fig.transFigure)
    fig.text(0.63, 0.85, s = 'b', **DEF_SUBLABEL, transform = fig.transFigure)
    fig.text(0.63, 0.5,  s = 'c', **DEF_SUBLABEL, transform = fig.transFigure)

    t_test_1 = df_MPL['distance'].iloc[-30:].tolist()
    t_test_2 = df_unique['distance'].tolist()    
    t_test_3 = df_func['distance'].iloc[-30:].tolist()
    bins = np.linspace(3, 27, 15)
    x = st.ttest_ind(a = t_test_1, b = t_test_2, equal_var = True)
    y = st.ttest_ind(a = t_test_3, b = t_test_2, equal_var = True)
    ax3.hist(t_test_2, bins, label = 'overall', density=True, alpha = 0.3, edgecolor = 'k')
    ax3.hist(t_test_1, bins, label = 'reference, t=%.2f p=%.2e'%(y[0],y[1]), density = True, alpha = 0.3, edgecolor = 'k')
    ax3.hist(t_test_3, bins, label = 'popDMS, t=%.2f p=%.2e'%(x[0],x[1]),    density = True, alpha = 0.3, edgecolor = 'k')
    ax3.set_xlabel('distance (Ångströms)', fontsize = SIZELABEL)
    ax3.set_ylabel('frequency',fontsize = SIZELABEL)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    
    legend_fig_4c = [Line2D([0], [0], color = 'lightskyblue',  lw = 2, label = 'overall', alpha = 1),
                     Line2D([0], [0], color = 'green', lw = 2, label = 'reference, t=%.2f p=%.2e'%(y[0],y[1]), alpha = 0.3),
                     Line2D([0], [0], color = 'orange',lw = 2, label = 'popDMS, t=%.2f p=%.2e'%(x[0],x[1]), alpha = 0.6),
                    ]
    ax3.legend(handles = legend_fig_4c, 
              loc = FIG4_C_LEGEND_XY, ncol = 1, 
              frameon = False, fontsize = SIZELABEL,
              handletextpad = 0.1)

    plt.savefig(FIG_DIR + 'Fig4_epistasis.pdf', dpi=200)
    plt.show() 


###

def SUPPFIG1_EPISTASIS():
    SUPPFIG1_RAWDATA_REP1 = './output/epistasis/tRNA/rep1_100.txt'
    SUPPFIG1_IDXMAT_REP1  = './output/epistasis/tRNA/index_matrix_rep1.csv'
    SUPPFIG1_RAWDATA_REP2 = './output/epistasis/tRNA/rep2_100.txt'
    SUPPFIG1_IDXMAT_REP2  = './output/epistasis/tRNA/index_matrix_rep2.csv'

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
    plt.savefig(FIG_DIR + 'Sup_Fig1_epistasis_correlation.pdf', dpi=200)
    plt.show()

