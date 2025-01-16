# -*- coding: utf-8 -*-
"""
Module Name: Fatigue damage comparison
Description: This module calculates the fatigue damage through different Spectral
methods and compares the results with RFC technique in TD.
"""

import rainflow
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.signal as sig
import Spectral_methods as fsm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import sys
from pathlib import Path


project_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_path))

# Now you can import from synthetic_data
from synthetic_data.Jonswap import jonswap_elevation


###############################################################################
'                         SN curve parameters from DNV                        '

def para_DNV(component):
    '''
    Function that gives the ad, m and gamma parameters according to DNV rule.
    for the selected chain type depending on the nominal diameter
    '''
    if component == "stud chain":
        ad = 1.2E11
        m = 3.0
        gamma = 1.3293
    elif component == "studless chain":
        ad = 6E10
        m = 3.0
        gamma = 1.3293
    elif component == "six strand rope":
        ad = 3.4E14
        m = 4
        gamma = 2
    elif component == "spiral strand rope":
        ad = 1.7E17
        m = 4.8
        gamma = 2.9812
    else:
        raise ValueError("Invalid component type")
    return m, gamma, ad

###############################################################################
'                                  TD damage                                  '
 
def RFC_damage(tension, t, k, C):
    cycles = rainflow.count_cycles(tension, nbins=200)
    D_list = []

    for bin in cycles:
        # Number of cycles to failure at tension range
        N = C / ((bin[0])**k)
        # Number of cycles within the tension range interval
        n = bin[1]
        D = n / N * 3600 * 24 * 365.25 / t[-1]
        D_list.append(D)
    D_RFC = sum(D_list)
    return D_RFC


###############################################################################
'                           Mooring characteristics                           '

Nominal_dia = 144 # mm (Taken from RM3 WEC-Sim values)
component = 'stud chain'
grade = 'R3'

# Calculate area based on component type and nominal diameter
if component == 'stud chain' or component == 'studless chain':
    Area = 2 * np.pi * (Nominal_dia / 2)**2
else:
    Area = np.pi * (Nominal_dia / 2)**2


###############################################################################
'                            Analysis characteristics                         '

ss = 70 # Sea State quantity
realisations = 50 # Number of realisations per ss
prb = 1 # Set all sea state occurrence probability as 1

###############################################################################

damage_data = {
    'RFC': [], 'NB': [], 'WL': [], 'OC': [], 'TB': [], 'A75': [], 'MTB':[],
    'DK': [], 'ZB': [], 'PK': [], 'JP': [], 'WU':[], 'JM': [], 'DNB': [], 
    'SO': [], 'FC': [], 'MFC': [], 'LWB': [], 'LOW': [], 'HMa':[], 'SM': [], 
    'LB': [], 'HM': [], 'BM': [], 'GZ':[], 'IBM':[]}

# Initialize dictionaries to store matrices for each method
damage_matrices = {}

for method in list(damage_data.keys()):
    damage_matrices[method] = np.zeros((realisations, ss))

path = '..\\synthetic_data'

LF           = np.load(f'{path}\\LF.npy')
LF_amplitude = np.load(f'{path}\\LF_amplitude.npy')

# Define time vector
t = np.linspace(0, 7500, 15000)
dt = t[1] - t[0]

# Define wave scatter diagram
points = int(np.sqrt(ss))
Tp_sd = np.linspace(4,18, points)
Hs_sd = np.linspace(0.1, 14, points)
sea_state = []

for hs in Hs_sd:
    for tp in Tp_sd:
        sea_state.append([hs, tp])
        
if len(sea_state) < ss:
    hs2 = np.linspace((sea_state[points][0]-sea_state[0][0])/2, 14 - (sea_state[points][0]-sea_state[0][0])/2, ss - len(sea_state))
    for z in hs2:
        sea_state.append([z, 11])

###############################################################################
'                               Damage estimation                             '

for i in tqdm(range(ss), desc="Processing", unit="iteration"): 
    
    
    for j in range(realisations):
    
        # Define sea state characteristics
        Hs = sea_state[i][0]
        Tp = sea_state[i][1]
    
        # Generate wave frequency time series
        tension_wave = jonswap_elevation(t, Hs, Tp, dt) * 3.5e3
        
        # Generate low frequency time series
        freq_low = LF[j,i]  
        Tp_LF = 1/freq_low
        Hs_LF = LF_amplitude[j,i]
        Tmean = 1.7e5
        tension_low = jonswap_elevation(t, Hs_LF, Tp_LF, dt) + Tmean
    
        # Generate the total time series
        tension = tension_wave + tension_low
        
        # Transform the tension into stress
        tension = np.array(tension) / Area
        tension_wave = np.array(tension_wave) / Area
        tension_low = np.array(tension_low) / Area
        
        dt = t[1] - t[0]
        fs = 1 / dt
        k, gamma, C = para_DNV(component)
    
        # Compute the PSD using Welch's method
        f, Pxx = sig.welch(tension, fs=fs, nperseg=4 * 1024)
        f_l, Pxx_l = sig.welch(tension_low, fs=fs, nperseg=4 * 1024)
        f_w, Pxx_w = sig.welch(tension_wave, fs=fs, nperseg=4 * 1024)
    
        # Compute the zero up crossing frequencies of the tension and low and wave 
        # tensions
        Mo = fsm.spectral_moments(f, Pxx)
        ZUCF = (1/(2*np.pi))*np.sqrt(Mo[2] / Mo[0])
        Mo_l = fsm.spectral_moments(f_l, Pxx_l)
        ZUCF_l = (1/(2*np.pi))*np.sqrt(Mo_l[2] / Mo_l[0])
        Mo_w = fsm.spectral_moments(f_w, Pxx_w)
        ZUCF_w = (1/(2*np.pi))*np.sqrt(Mo_w[2] / Mo_w[0])
    
##################################### TD ######################################
        
    # Calculate rainflow damage
        D_rainflow = RFC_damage(tension, t, k, C) * prb
        damage_matrices['RFC'][j,i] = D_rainflow

     
############################## Spectral methods ###############################
    
    # Narrowband
        D_nb = fsm.Narrowband(k, C, Mo, ZUCF) * 3.15576E7 * prb
        damage_matrices['NB'][j,i] = D_nb
        
    # Jiao Moan
        D_jm = fsm.Jiao_Moan(k, ZUCF, Mo_w, Mo_l, ZUCF_l, ZUCF_w) * D_nb
        damage_matrices['JM'][j,i] = D_jm  
        
    # Dual narrowband
        D_dnb = fsm.dual_narrowband(k, ZUCF, Mo_w, Mo_l, ZUCF_l, ZUCF_w) * D_nb
        damage_matrices['DNB'][j,i] = D_dnb
    
    # Sakai Okamura
        D_so = fsm.Sakai_Okamura(k, C, Mo_l, Mo_w, ZUCF_l, ZUCF_w) * 3.15576E7 * prb
        damage_matrices['SO'][j,i] = D_so  
    
    # Fu Cebon
        D_fc = fsm.Fu_Cebon(k, C, Mo_w, Mo_l, ZUCF_w, ZUCF_l, solution = 'Binomial expansion')* 3.15576E7 * prb
        damage_matrices['FC'][j,i] = D_fc 
    
    # Modified Fu Cebon
        D_mfc = fsm.Modified_Fu_Cebon(k, C, Mo_w, Mo_l, ZUCF_w, ZUCF_l, solution = 'Binomial expansion')* 3.15576E7 * prb
        damage_matrices['MFC'][j,i] = D_mfc
    
    # Low Bimodal
        D_lwb = fsm.LowBimodal(k, C, ZUCF_w, ZUCF_l, Mo_w, Mo_l)* 3.15576E7 * prb
        damage_matrices['LWB'][j,i] = D_lwb    
    
    # Low (2014)
        D_low = fsm.Low(k, C, Mo, Mo_l, Mo_w, ZUCF, ZUCF_l, ZUCF_w)* 3.15576E7 * prb
        damage_matrices['LOW'][j,i] = D_low
    
    # Han Ma
        D_hma = fsm.Han_Ma(k, C, ZUCF, Mo_w, Mo_l, ZUCF_l, ZUCF_w)* 3.15576E7 * prb
        damage_matrices['HMa'][j,i] = D_hma
        
    # Lotsberg
        D_lb = fsm.Lotsberg(k, C, Mo_w, Mo_l, ZUCF_w, ZUCF_l)* 3.15576E7 * prb
        damage_matrices['LB'][j,i] = D_lb
    
    # Dirlik
        D_dk = fsm.Dirlik(Mo, k, C)* 3.15576E7 * prb
        damage_matrices['DK'][j,i] = D_dk
    
    # Zhao Baker
        D_zb = fsm.Zhao_Baker(k, C, Mo, Pxx, f)* 3.15576E7 * prb
        damage_matrices['ZB'][j,i] = D_zb
        
    # Park
        D_pk = fsm.Park(Mo, k, C, f, Pxx)* 3.15576E7 * prb
        damage_matrices['PK'][j,i] = D_pk
        
    # Jun Park
        D_jp = fsm.Jun_Park(k, C, Mo, f, Pxx)* 3.15576E7 * prb
        damage_matrices['JP'][j,i] = D_jp
        
    # Wu method
        D_wu = fsm.wu(k, C, Mo)* 3.15576E7 * prb
        damage_matrices['WU'][j,i] = D_wu
    
    # Huang-Moan
        D_hm = fsm.Huang_Moan(k, C, Mo_w, Mo_l, ZUCF_w, ZUCF_l)* 3.15576E7 * prb
        damage_matrices['HM'][j,i] = D_hm
        
    # Wirsching Light
        rho_WL = fsm.Wirsching_Light(k, Mo)
        D_wl = D_nb * rho_WL
        damage_matrices['WL'][j,i] = D_wl
    
    # Tovo Benasciutti
        rho_TB= fsm.Tovo_Benasciutti(Mo, k, tension)
        D_tb = D_nb * rho_TB
        damage_matrices['TB'][j,i] = D_tb
        
    # Alfa 0.75
        rho075 = fsm.alfa075(Mo, f, Pxx)
        D_a75 = D_nb * rho075
        damage_matrices['A75'][j,i] = D_a75
        
    # Ortiz Chen
        rho_OC = fsm.Ortiz_Chen(k, Mo, f, Pxx)
        D_oc = D_nb * rho_OC
        damage_matrices['OC'][j,i] = D_oc
    
    # Modified Tovo Benasciutti
        rho_MTB= fsm.Modified_Tovo_Benasciutti(Mo, k, tension)
        D_mtb = D_nb * rho_MTB
        damage_matrices['MTB'][j,i] = D_mtb
    
    # Single moment
        D_sm = fsm.SingleMoment(k, C, f, Pxx)* 3.15576E7 * prb
        damage_matrices['SM'][j,i] = D_sm
    
    # Bands method
        D_bm = fsm.BandsMethod(k, C, tension, fs, nbands=200)* 3.15576E7 * prb
        damage_matrices['BM'][j,i] = D_bm
    
    # Gao Zheng
        D_gz = fsm.Gao_Zheng(k, C, f_l, Pxx_l, f_w, Pxx_w, ZUCF_w, ZUCF_l, ZUCF)* 3.15576E7 * prb
        damage_matrices['GZ'][j,i] = D_gz
        
    # Improved Bands method
        D_ibm = fsm.ImprovedBandsMethod(k, C, tension_low, tension_wave, fs, 200, Mo_w, Mo_l)* 3.15576E7 * prb
        damage_matrices['IBM'][j,i] = D_ibm
        
        
    # Calculate mean damages and append to corresponding lists
    for key, value in damage_data.items():
        method_list = damage_matrices[f'{key}'][:,i]
        value.append(np.mean(method_list))
        

damage_df = pd.DataFrame(damage_data)

# Normalise damage values
norm_damage = damage_df.div(damage_df['RFC'], axis=0)
norm_damage = norm_damage.drop(columns=['RFC'])

# List of methods (column names)
methods = list(norm_damage.keys())

# Get boxplot values for the normalised damage
norm_boxplot = pd.concat([norm_damage.median(), norm_damage.mean(), norm_damage.quantile(0.25), norm_damage.quantile(0.75)], axis = 1)
norm_boxplot = norm_boxplot.rename(columns={0: "median", 1: "mean", 0.25: "Q1", 0.75:"Q3"})

# Error index (Benasciutti and Tovo)
error_index = np.sqrt(np.log10(norm_damage).pow(2).sum() / ss)

# Relative damage difference
reldif_damage = abs(damage_df.subtract(damage_df['RFC'], axis=0)).div(damage_df['RFC'], axis=0)
reldif_damage = reldif_damage.drop(columns=['RFC'])

# Get boxplot values for the relative damage difference
reldif_boxplot = pd.concat([reldif_damage.median(), reldif_damage.mean(), reldif_damage.quantile(0.25), reldif_damage.quantile(0.75)], axis = 1)
reldif_boxplot = reldif_boxplot.rename(columns={0: "median", 1: "mean", 0.25: "Q1", 0.75:"Q3"})


# Normalised damage boxplot
fig = plt.figure(figsize=(12, 6))
ax = fig.add_axes([0, 0, 1, 1])
bp = ax.boxplot(norm_damage)
ax.set_xticklabels(methods, fontsize=16)
plt.yticks(fontsize=14)
plt.ylabel('Normalised damage', fontsize=16)
# Add a constant line at y=1
ax.axhline(y=1, color='green')
plt.ylim([-0,3])
plt.show()
# fig.savefig('Results\\Normalised_damage_comparison_3.svg', dpi=300, bbox_inches='tight')

# Relative damage difference boxplot
fig = plt.figure(figsize=(12, 6))
ax = fig.add_axes([0, 0, 1, 1])
bp = ax.boxplot(reldif_damage)
ax.set_xticklabels(methods, fontsize=16)
plt.yticks(fontsize=14)
plt.ylabel('Damage difference', fontsize=16)
# plt.ylim([0,2])
plt.show()
# fig.savefig('Results\\Damage_difference_comparison_3.svg', dpi=300, bbox_inches='tight')

method_colors = {
    'RFC': '#33CCCCff', 'NB': '#33CCCCff', 'WL': '#009990ff', 'OC': '#006969ff',
    'TB': '#717171ff', 'A75': '#000000ff', 'MTB': '#33CCCCff', 'DK': '#717171ff', 
    'ZB': '#009990ff', 'PK': '#006969ff', 'JP': '#717171ff', 'WU': '#006969ff', 
    'JM': '#000000ff', 'DNB': '#33CCCCff', 'SO': '#009990ff', 'FC': '#006969ff', 
    'MFC': '#717171ff', 'LWB': '#000000ff', 'LOW': '#33CCCCff', 'HMa': '#009990ff', 
    'SM': '#009990ff', 'LB': '#006969ff', 'HM': '#717171ff', 'BM': '#000000ff', 
    'GZ': '#717171ff', 'IBM': '#33CCCCff'
}

method_markers = {
    'RFC': '-', 'NB': 'o', 'WL': 's', 'OC': 'D', 'TB': 'v', 'A75': ',', 'MTB': 'x',
    'DK': '+', 'ZB': '*', 'PK': '^', 'WU': 'o', 'JP': '<', 'JM': 'o', 'DNB': 's',
    'SO': 'd', 'FC': 'v', 'MFC': ',', 'LWB': '+', 'LOW': '*','HMa': 'x', 'SM': '^',
    'LB': '<', 'HM': 'o', 'BM': '.', 'GZ': 'x', 'IBM': '.'
}

method_marker_face = {
    'RFC': '#33CCCCff', 'NB': '#33CCCCff', 'WL': '#009990ff', 'OC': 'none',
    'TB': '#717171ff', 'A75': 'none', 'MTB': '#33CCCCff', 'DK': '#717171ff', 
    'ZB': '#009990ff', 'PK': '#006969ff', 'JP': 'none', 'WU': 'none', 
    'JM': '#000000ff', 'DNB': 'none', 'SO': '#009990ff', 'FC': 'none', 
    'MFC': 'none', 'LWB': '#000000ff', 'LOW': '#33CCCCff', 'HMa': '#009990ff', 
    'SM': '#009990ff', 'LB': '#006969ff', 'HM': 'none', 'BM': '#000000ff', 
    'GZ': '#717171ff', 'IBM': '#33CCCCff'
}



methods_a = ['WL', 'OC', 'TB', 'A75', 'MTB']
methods_b = ['DK', 'ZB', 'PK', 'JP', 'WU']
methods_c = ['JM', 'DNB', 'SO', 'FC', 'MFC', 'LWB', 'LOW', 'HMa']
methods_d = ['SM', 'LB', 'HM', 'BM', 'GZ', 'IBM']

RFC=list(damage_df.RFC)
RFC.append(5e-4) # Elongates the RFC damage so that it can be plotted in a square graph

# Damage scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
plt.plot(RFC, RFC, label='RFC', color = '#000000ff', linewidth = 2)
# Loop through methods and create scatter plots
for method in damage_df.columns[1:]:
    plt.scatter(damage_df.RFC, damage_df[method], label=method, color=method_colors[method], marker=method_markers[method], facecolors=method_marker_face[method])

plt.xlabel('Fatigue damage - Time domain', fontsize = 16)
plt.ylabel('Fatigue damage - Frequency domain', fontsize = 16)
plt.xlim([0, 5e-4])
plt.ylim([0, 5e-4])
ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
plt.legend(fontsize = 14, loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
# fig.savefig('Results\\Damage_comparison.svg', dpi=300, bbox_inches='tight')

group_list = [methods_a, methods_b, methods_c, methods_d]

for n in range(0,4):   
    # Damage scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    plt.plot(RFC, RFC, label='RFC', color = '#000000ff', linewidth = 2)
    # Loop through methods and create scatter plots
    for method in group_list[n]:
        plt.scatter(damage_df.RFC, damage_df[method], label=method, color=method_colors[method], marker=method_markers[method], facecolors=method_marker_face[method])
    
    plt.xlabel('Fatigue damage - Time domain', fontsize = 16)
    plt.ylabel('Fatigue damage - Frequency domain', fontsize = 16)
    plt.xlim([0, 5e-4])
    plt.ylim([0, 5e-4])
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    plt.legend(fontsize = 14, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    # fig.savefig(f'Results\\Damage_comparison_{n}.svg', dpi=300, bbox_inches='tight')


