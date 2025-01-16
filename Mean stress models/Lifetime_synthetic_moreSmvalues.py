"""
Module Name: 
Description: 
Author: Eguzkiñe Martinez Puente
Created on: 13-05-2024
"""

import rainflow
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
from common_functions import *

import sys
from pathlib import Path


project_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_path))

# Now you can import from synthetic_data
from synthetic_data.Jonswap import jonswap_elevation



###############################################################################
'                                  TD damage                                  '
 

def RFC_damage(tension, t, k, C, Sm_correction=None, Su=None, Sy=None, Sf=None, alphak=1, Sm=None):
    # Extract rainflow cycles and convert to NumPy arrays for efficiency
    cycles = np.array(list(rainflow.extract_cycles(tension)))
    amplitude = cycles[:, 0] / 2
    Smean = cycles[:, 1]
    counts = cycles[:, 2]

    # Precompute constants
    time_factor = 1 / t[-1] * 3600 * 24 * 365.25
    if Sm is None:
        Sm = np.mean(tension)
    
    # Mean stress correction functions as a lookup dictionary
    correction_methods = {
        None: lambda Sa, Sm: Sa,
        'SWT': lambda Sa, Sm: Sa * Smith_Watson_Topper(Sa, Sm + Sa),
        'Goodman': lambda Sa, Sm: Sa * Goodman(Sm, Su),
        'Gerber': lambda Sa, Sm: Sa * Gerber(Sm, Su),
        'Soderberg': lambda Sa, Sm: Sa * Soderberg(Sm, Sy),
        'Kwofie': lambda Sa, Sm: Sa * Kwofie(Sm, Su, alphak),
        'Morrow': lambda Sa, Sm: Sa * Morrow(Sm, Sf),
    }
    
    # Apply the selected mean stress correction
    Sar = np.array([correction_methods[Sm_correction](Sa, Sm) for Sa in amplitude])

    # Calculate number of cycles to failure (N) and damage
    N = C / (2 * Sar) ** k
    D_list = counts / N * time_factor

    # Total damage and equivalent stress
    d_RFC = np.sum(D_list)
    n_ss = np.sum(counts * time_factor)
    s_eq = np.mean(2 * Sar)

    return d_RFC, n_ss, s_eq


###############################################################################
'                           RM3 lifetime computation                          '

Nominal_dia = 144 # mm
component   = 'studless chain'
grade       = 'R3'
Su, Sy, MBL = Properties(grade, Nominal_dia).values()
k, gamma, C = para_DNV(component)


if grade == 'R3':
    # Selected data R3
    CSWT, CGOODMAN, CGERBER, CSODERBERG = 1.32E17, 1.03E11, 6.51E10, 1.49E11 
    CKWOFIE, CMORROW = 9.83E10, 6.62E10 # Obtained after correcting DNV test data to R=-1
    kswt = 5.5
    Sf = 2022
elif grade == 'R4':
    # Selected data R4
    CSWT, CGOODMAN, CGERBER, CSODERBERG = 2.11E17, 1.03E11, 6.51E10, 1.49E11 
    CKWOFIE, CMORROW = 1.03E11, 6.62E10 # Obtained after correcting DNV test data to R=-1
    kswt = 5.5
    Sf = 2022
else:
    print('There is no data available for the selected material.')



if component == 'stud chain' or component == 'studless chain':
    Area = 2*np.pi*Nominal_dia**2/4
else:
    Area = np.pi*Nominal_dia**2/4


MBL_percentage = [0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]

# Initialize lists
D_DNV, S_RF, N_RF = [], [], []
D_SWT, S_RF_SWT = {Sm_p: [] for Sm_p in MBL_percentage}, {Sm_p: [] for Sm_p in MBL_percentage}
D_GOODMAN, S_RF_GOODMAN = {Sm_p: [] for Sm_p in MBL_percentage}, {Sm_p: [] for Sm_p in MBL_percentage}
D_GERBER, S_RF_GERBER = {Sm_p: [] for Sm_p in MBL_percentage}, {Sm_p: [] for Sm_p in MBL_percentage}
D_SODERBERG, S_RF_SODERBERG = {Sm_p: [] for Sm_p in MBL_percentage}, {Sm_p: [] for Sm_p in MBL_percentage}
D_KWOFIE, S_RF_KWOFIE = {Sm_p: [] for Sm_p in MBL_percentage}, {Sm_p: [] for Sm_p in MBL_percentage}
D_MORROW, S_RF_MORROW = {Sm_p: [] for Sm_p in MBL_percentage}, {Sm_p: [] for Sm_p in MBL_percentage}
# L_DNV = []



###############################################################################
'                            Analysis characteristics                         '

ss = 64 
realisations = 1#50 
prb = 1 # Set all sea state occurrence probability as 1

###############################################################################


    
path = '..\\synthetic_data'

LF           = np.load(f'{path}\\LF.npy')
LF_amplitude = np.load(f'{path}\\LF_amplitude.npy')

# Define time vector
t   = np.linspace(0, 7500, 15000)
dt  = t[1] - t[0]
fs  = 1/dt

# Define wave scatter diagram
points    = int(np.sqrt(ss))
Tp_sd     = np.linspace(4, 18, points)
Hs_sd     = np.linspace(0.1, 14, points)
sea_state = []

for hs in Hs_sd:
    for tp in Tp_sd:
        sea_state.append([hs, tp])

Hs_list = list(Hs_sd)
Tp_list = list(Tp_sd)
if len(sea_state) < ss:
    hs2 = np.linspace((sea_state[points][0]-sea_state[0][0])/2, 14 - 
                      (sea_state[points][0]-sea_state[0][0])/2, ss - len(sea_state))
    for z in hs2:
        sea_state.append([z, 11])
        # Hs_list.append(z)
# Tp_list.append(11)

index = []

for i in range(points):
    for j in range(points):
        index.append([j, i])
# dif = ss - len(index)
# for i in range(dif):
#     index.append([i+points, 8])

rho         = 1025 # kg/m3 for seawater
g           = 9.81 # m/s2
Energy_ss   = []

for ii in range(len(sea_state)):
    Energy_ss.append(1/64/np.pi*rho*g**2*sea_state[ii][0]**2*sea_state[ii][1])

###############################################################################
'                               Damage estimation                             '

reldif_damage_10 = {'SWT': np.zeros((8,8)),'GOODMAN': np.zeros((8,8)), 'GERBER': np.zeros((8,8)),
                    'SODERBERG': np.zeros((8,8)), 'MORROW': np.zeros((8,8)), 'KWOFIE': np.zeros((8,8))}
reldif_damage_20 = {'SWT': np.zeros((8,8)),'GOODMAN': np.zeros((8,8)), 'GERBER': np.zeros((8,8)),
                    'SODERBERG': np.zeros((8,8)), 'MORROW': np.zeros((8,8)), 'KWOFIE': np.zeros((8,8))}
reldif_damage_30 = {'SWT': np.zeros((8,8)),'GOODMAN': np.zeros((8,8)), 'GERBER': np.zeros((8,8)),
                    'SODERBERG': np.zeros((8,8)), 'MORROW': np.zeros((8,8)), 'KWOFIE': np.zeros((8,8))}
reldif_damage_15 = {'SWT': np.zeros((8,8)),'GOODMAN': np.zeros((8,8)), 'GERBER': np.zeros((8,8)),
                    'SODERBERG': np.zeros((8,8)), 'MORROW': np.zeros((8,8)), 'KWOFIE': np.zeros((8,8))}
reldif_damage_25 = {'SWT': np.zeros((8,8)),'GOODMAN': np.zeros((8,8)), 'GERBER': np.zeros((8,8)),
                    'SODERBERG': np.zeros((8,8)), 'MORROW': np.zeros((8,8)), 'KWOFIE': np.zeros((8,8))}
reldif_damage_12 = {'SWT': np.zeros((8,8)),'GOODMAN': np.zeros((8,8)), 'GERBER': np.zeros((8,8)),
                    'SODERBERG': np.zeros((8,8)), 'MORROW': np.zeros((8,8)), 'KWOFIE': np.zeros((8,8))}
reldif_damage_17 = {'SWT': np.zeros((8,8)),'GOODMAN': np.zeros((8,8)), 'GERBER': np.zeros((8,8)),
                    'SODERBERG': np.zeros((8,8)), 'MORROW': np.zeros((8,8)), 'KWOFIE': np.zeros((8,8))}
reldif_damage_22 = {'SWT': np.zeros((8,8)),'GOODMAN': np.zeros((8,8)), 'GERBER': np.zeros((8,8)),
                    'SODERBERG': np.zeros((8,8)), 'MORROW': np.zeros((8,8)), 'KWOFIE': np.zeros((8,8))}
reldif_damage_27 = {'SWT': np.zeros((8,8)),'GOODMAN': np.zeros((8,8)), 'GERBER': np.zeros((8,8)),
                    'SODERBERG': np.zeros((8,8)), 'MORROW': np.zeros((8,8)), 'KWOFIE': np.zeros((8,8))}

for Sm_p in MBL_percentage:

    damage_data = {'DNV': np.zeros((8,8)),'SWT': np.zeros((8,8)),'GOODMAN': np.zeros((8,8)),'GERBER': np.zeros((8,8)),
                        'SODERBERG': np.zeros((8,8)), 'MORROW': np.zeros((8,8)), 'KWOFIE': np.zeros((8,8))}
    S_data = {'DNV': np.zeros((8,8)),'SWT': np.zeros((8,8)),'GOODMAN': np.zeros((8,8)), 'GERBER': np.zeros((8,8)),
                        'SODERBERG': np.zeros((8,8)), 'MORROW': np.zeros((8,8)), 'KWOFIE': np.zeros((8,8))}
    # Initialize dictionaries to store matrices for each method
    damage_matrices = {}
    S_matrices = {}
    N_matrix = np.zeros((realisations, ss))

    for method in list(damage_data.keys()):
        damage_matrices[method] = np.zeros((realisations, ss))
        S_matrices[method] = np.zeros((realisations, ss))
        
    Sm  = Sm_p*MBL*1000 / Area
    ii   = 0 
    for i in range(ss): 
        
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
    
    ##################################### TD ######################################
            
        # Calculate rainflow damage
            # According to DNV
            D_dnv, n_ss, s_eq                   = RFC_damage(tension, t, k, C, Sm=Sm) 
            damage_matrices['DNV'][j,i]         = D_dnv * prb
            S_matrices['DNV'][j,i]              = s_eq
            N_matrix[j,i]                       = n_ss * prb
            
            D_swt, _ , s_eq                     = RFC_damage(tension, t, kswt, CSWT, Sm_correction='SWT', Sm=Sm)
            damage_matrices['SWT'][j,i]         = D_swt * prb
            S_matrices['SWT'][j,i]              = s_eq
            
            D_goodman, _ , s_eq                 = RFC_damage(tension, t, k, CGOODMAN, Sm_correction='Goodman', Su = Su, Sm=Sm)
            damage_matrices['GOODMAN'][j,i]     = D_goodman * prb
            S_matrices['GOODMAN'][j,i]          = s_eq
            
            D_gerber, _ , s_eq                  = RFC_damage(tension, t, k, CGERBER, Sm_correction='Gerber', Su = Su, Sm=Sm)
            damage_matrices['GERBER'][j,i]      = D_gerber * prb
            S_matrices['GERBER'][j,i]           = s_eq
            
            D_soderberg, _ , s_eq               = RFC_damage(tension, t, k, CSODERBERG, Sm_correction='Soderberg', Sy = Sy, Sm=Sm)
            damage_matrices['SODERBERG'][j,i]   = D_soderberg * prb
            S_matrices['SODERBERG'][j,i]        = s_eq
            
            D_morrow, _ , s_eq                  = RFC_damage(tension, t, k, CMORROW, Sm_correction='Morrow', Sf = 2*Sf, Sm=Sm)
            damage_matrices['MORROW'][j,i]      = D_morrow * prb
            S_matrices['MORROW'][j,i]           = s_eq
            
            D_kwofie, _ , s_eq                  = RFC_damage(tension, t, k, CKWOFIE, Sm_correction='Kwofie', Su = Su, Sm=Sm)
            damage_matrices['KWOFIE'][j,i]      = D_kwofie * prb
            S_matrices['KWOFIE'][j,i]           = s_eq
         
            
        # Calculate mean damages and append to corresponding lists
        for key, value in damage_data.items():
            method_list                                     = damage_matrices[f'{key}'][:,i]
            damage_data[f'{key}'][index[i][1],index[i][0]]  = np.mean(method_list)
            S_list                                          = S_matrices[f'{key}'][:,i]
            S_data[f'{key}'][index[i][1],index[i][0]]       = np.mean(S_list)
 
        if Sm_p == 0.1:    
            D_DNV.append(damage_data['DNV'][index[i][1],index[i][0]])
            S_RF.append(S_data['DNV'][index[i][1],index[i][0]])
            N_RF.append(np.mean(N_matrix[:,i]))
            
        D_SWT[Sm_p].append(damage_data['SWT'][index[i][1],index[i][0]])
        S_RF_SWT[Sm_p].append(S_data['SWT'][index[i][1],index[i][0]])
        D_GOODMAN[Sm_p].append(damage_data['GOODMAN'][index[i][1],index[i][0]])
        S_RF_GOODMAN[Sm_p].append(S_data['GOODMAN'][index[i][1],index[i][0]])
        D_GERBER[Sm_p].append(damage_data['GERBER'][index[i][1],index[i][0]])
        S_RF_GERBER[Sm_p].append(S_data['GERBER'][index[i][1],index[i][0]])
        D_SODERBERG[Sm_p].append(damage_data['SODERBERG'][index[i][1],index[i][0]])
        S_RF_SODERBERG[Sm_p].append(S_data['SODERBERG'][index[i][1],index[i][0]])
        D_MORROW[Sm_p].append(damage_data['MORROW'][index[i][1],index[i][0]])
        S_RF_MORROW[Sm_p].append(S_data['MORROW'][index[i][1],index[i][0]])
        D_KWOFIE[Sm_p].append(damage_data['KWOFIE'][index[i][1],index[i][0]])
        S_RF_KWOFIE[Sm_p].append(S_data['KWOFIE'][index[i][1],index[i][0]])
        
        # Copy values from the original dictionary to the new one and normalize
        for key in damage_data.keys():
            if key != 'DNV':
                if Sm_p == 0.1:
                    for i in range(len(damage_data[key])):
                        reldif_damage_10[key] = (damage_data[key] - damage_data['DNV']) / damage_data['DNV'] * 100
                elif Sm_p == 0.2:
                    for i in range(len(damage_data[key])):
                        reldif_damage_20[key] = (damage_data[key] - damage_data['DNV']) / damage_data['DNV'] * 100
                elif Sm_p == 0.3:
                    for i in range(len(damage_data[key])):
                        reldif_damage_30[key] = (damage_data[key] - damage_data['DNV']) / damage_data['DNV'] * 100
                elif Sm_p == 0.15:
                    for i in range(len(damage_data[key])):
                        reldif_damage_15[key] = (damage_data[key] - damage_data['DNV']) / damage_data['DNV'] * 100
                elif Sm_p == 0.25:
                    for i in range(len(damage_data[key])):
                        reldif_damage_25[key] = (damage_data[key] - damage_data['DNV']) / damage_data['DNV'] * 100
                elif Sm_p == 0.12:
                    for i in range(len(damage_data[key])):
                        reldif_damage_12[key] = (damage_data[key] - damage_data['DNV']) / damage_data['DNV'] * 100
                elif Sm_p == 0.17:
                    for i in range(len(damage_data[key])):
                        reldif_damage_17[key] = (damage_data[key] - damage_data['DNV']) / damage_data['DNV'] * 100
                elif Sm_p == 0.22:
                    for i in range(len(damage_data[key])):
                        reldif_damage_22[key] = (damage_data[key] - damage_data['DNV']) / damage_data['DNV'] * 100
                elif Sm_p == 0.27:
                    for i in range(len(damage_data[key])):
                        reldif_damage_27[key] = (damage_data[key] - damage_data['DNV']) / damage_data['DNV'] * 100

        
   
damage_data = {
    'SWT': D_SWT,
    'GOODMAN': D_GOODMAN,
    'GERBER': D_GERBER,
    'SODERBERG': D_SODERBERG,
    'KWOFIE': D_KWOFIE,
    'MORROW': D_MORROW
    }

S_equivalent = {
    'SWT': S_RF_SWT,
    'GOODMAN': S_RF_GOODMAN,
    'GERBER': S_RF_GERBER,
    'SODERBERG': S_RF_SODERBERG,
    'KWOFIE': S_RF_KWOFIE,
    'MORROW': S_RF_MORROW
    }


total_damage_DNV    = sum(D_DNV)
Lifetime_DNV        = 1/total_damage_DNV
N_data_DNV          = [n * Lifetime_DNV for n in N_RF]
N_total_DNV         = sum(N_data_DNV)
S_total_DNV         = (C/N_total_DNV)**(1/k)

damage_df           = pd.DataFrame(damage_data)

total_damage = {
    'SWT': {Sm_p: 0 for Sm_p in MBL_percentage},
    'GOODMAN': {Sm_p: 0 for Sm_p in MBL_percentage},
    'GERBER': {Sm_p: 0 for Sm_p in MBL_percentage},
    'SODERBERG': {Sm_p: 0 for Sm_p in MBL_percentage},
    'KWOFIE': {Sm_p: 0 for Sm_p in MBL_percentage},
    'MORROW': {Sm_p: 0 for Sm_p in MBL_percentage}
    
    }

Lifetime = {
    'SWT': {Sm_p: 0 for Sm_p in MBL_percentage},
    'GOODMAN': {Sm_p: 0 for Sm_p in MBL_percentage},
    'GERBER': {Sm_p: 0 for Sm_p in MBL_percentage},
    'SODERBERG': {Sm_p: 0 for Sm_p in MBL_percentage},
    'KWOFIE': {Sm_p: 0 for Sm_p in MBL_percentage},
    'MORROW': {Sm_p: 0 for Sm_p in MBL_percentage}
    }

N_data = {
    'SWT': {Sm_p: [] for Sm_p in MBL_percentage},
    'GOODMAN': {Sm_p: [] for Sm_p in MBL_percentage},
    'GERBER': {Sm_p: 0 for Sm_p in MBL_percentage},
    'SODERBERG': {Sm_p: 0 for Sm_p in MBL_percentage},
    'KWOFIE': {Sm_p: 0 for Sm_p in MBL_percentage},
    'MORROW': {Sm_p: 0 for Sm_p in MBL_percentage}
    }

N_total = {
    'SWT': {Sm_p: 0 for Sm_p in MBL_percentage},
    'GOODMAN': {Sm_p: 0 for Sm_p in MBL_percentage},
    'GERBER': {Sm_p: 0 for Sm_p in MBL_percentage},
    'SODERBERG': {Sm_p: 0 for Sm_p in MBL_percentage},
    'KWOFIE': {Sm_p: 0 for Sm_p in MBL_percentage},
    'MORROW': {Sm_p: 0 for Sm_p in MBL_percentage}
    }

S_total = {
    'SWT': {Sm_p: 0 for Sm_p in MBL_percentage},
    'GOODMAN': {Sm_p: 0 for Sm_p in MBL_percentage},
    'GERBER': {Sm_p: 0 for Sm_p in MBL_percentage},
    'SODERBERG': {Sm_p: 0 for Sm_p in MBL_percentage},
    'KWOFIE': {Sm_p: 0 for Sm_p in MBL_percentage},
    'MORROW': {Sm_p: 0 for Sm_p in MBL_percentage}
    }


plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16
})

print(f"Max fatigue life DNV: \t \t \t \t \t {str(round(Lifetime_DNV))} years")
for key in total_damage.keys():
    for val in MBL_percentage:
        total_damage[key][val]  = sum(damage_df[key][val])
        Lifetime[key][val]      = 1/total_damage[key][val]
        N_data[key][val]        = [n * Lifetime[key][val] for n in N_RF]
        N_total[key][val]       = sum(N_data[key][val])
        if key == 'SWT': S_total[key][val]       = (CSWT/N_total[key][val])**(1/kswt)
        else: S_total[key][val]       = (CGOODMAN/N_total[key][val])**(1/k)
        tab = '\t'
        if key == 'SWT': tab = '\t \t'
        print(f"Max fatigue life {key} at {val}MBL: {tab} {str(round(Lifetime[key][val]))} years")


method_colors = {'DNV': '#009990ff', 'SWT': '#717171ff', 'GOODMAN': '#006969ff',
    'GERBER': '#000000ff', 'SODERBERG': '#006969ff', 'MORROW': '#717171ff',
    'KWOFIE': '#000000ff'}

method_line = {'DNV': '-', 'SWT': '-', 'GOODMAN': '-', 'GERBER': '-',
    'SODERBERG': '-.', 'MORROW': ':', 'KWOFIE': ':' }

method_markers  = {'DNV': 'o', 'SWT': 's', 'GOODMAN': 'D'}

# Plot: Lifetime vs Mean stress percentage
fig = plt.figure()
plt.plot(MBL_percentage, [1]*len(MBL_percentage), label = 'DNV', color = method_colors['DNV'], linestyle = method_line['DNV'])
for key in Lifetime:
    Norm_life = [val/Lifetime_DNV for val in Lifetime[key].values()]
    plt.plot(MBL_percentage, Norm_life, label = key, color = method_colors[key], linestyle = method_line[key])
plt.xlabel('Mean load [% MBL]')
plt.ylabel('Normalised lifetime [-]')
plt.yscale('log')
plt.legend()
plt.show()


# Data for plotting (example based on structure of your script)
MBL_percentage = [10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30]  # Mean load percentages
methods = ['GOODMAN', 'SWT', 'SODERBERG', 'GERBER', 'MORROW', 'KWOFIE']

# Extract normalized lifetime values
normalized_lifetimes = {method: [] for method in methods}
for method in methods:
    for mbl in MBL_percentage:
        if mbl / 100 in Lifetime[method]:
            normalized_lifetime = Lifetime[method][mbl / 100] / Lifetime_DNV * 100
            normalized_lifetimes[method].append(normalized_lifetime)
        else:
            normalized_lifetimes[method].append(None)

# Colors and markers for methods
method_colors = {
    'GOODMAN': '#33CCCC',
    'SWT': '#717171',
    'SODERBERG': '#006969',
    'GERBER': '#000000',
    'MORROW': '#717171',
    'KWOFIE': '#000000'
}
method_markers = {
    'GOODMAN': 'o',
    'SWT': 's',
    'SODERBERG': 'D',
    'GERBER': '^',
    'MORROW': 'v',
    'KWOFIE': 'x'
}

# Create the plot
fig, ax = plt.subplots(figsize=(8, 5))

# Add green and red zones
ax.axvspan(10, 23.7, facecolor='#CCFFCC', alpha=0.5, label="Conservative zone")
ax.axvspan(23.7, 30, facecolor='#FFCCCC', alpha=0.5, label="Risk zone")

# Plot the normalized lifetime for each method
for method in methods:
    ax.plot(
        MBL_percentage,
        normalized_lifetimes[method],
        label=method,
        color=method_colors[method],
        marker=method_markers[method],
        linestyle=':'
    )

# Add vertical dashed line to separate zones
ax.axvline(x=20, color='black', linestyle='--', linewidth=1)

# Add a horizontal line at 100
ax.axhline(y=100, color='gray', linestyle='-', linewidth=1)

# Labels and title
ax.set_xlabel('Mean load [%MBL]', fontsize=12)
ax.set_ylabel('Normalised lifetime [%]', fontsize=12)

# Add legend
ax.legend(loc='upper right')

# Adjust limits
ax.set_xlim(10, 30)
ax.set_ylim(50, 1500)
ax.set_yscale('log')

# Set specific tick positions and labels for the y-axis
ax.set_yticks([50, 500])
ax.set_yticklabels(['50', '500'])

# Show the plot
plt.tight_layout()
plt.show()
