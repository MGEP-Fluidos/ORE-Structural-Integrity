import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import ScalarFormatter
from common_functions import *

# Plotting configurations
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman'

# Define the power-law function and R² calculation
def power_law(x, a, b):
    return a * np.power(x, b)

def calculate_r2(y, y_fit):
    residuals = y - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - (ss_res / ss_tot)


###############################################################################
# Gabrielsen et al. Data
###############################################################################
data = {
    'Mean load [%MBL]': [8, 8, 8, 8, 8, 18, 18, 18, 18, 18],
    'Load range [MPa]': [65, 65, 60, 57, 55, 55, 50, 45, 40, 38],
    'Cycles to failure': [274712, 455958, 710927, 1189289, 1334048, 238142, 355267, 354534, 802907, 1455906],
    'Break location': ['Crown', 'Straight', 'Crown', 'Crown', 'Crown', 'Crown', 'Crown', 'Crown', 'Crown', 'Straight']
}
df = pd.DataFrame(data, index=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10'])

# Material properties
material = "R4"
d_nom = 114  # mm
Area = 2 * (np.pi * d_nom**2 / 4)
Su, Sy, MBL = Properties(material, d_nom).values()
alpha = 1
Sfp = 3915

# Calculate stress values
df['Sa [MPa]'] = df['Load range [MPa]'] / 2
df['Sm [MPa]'] = df['Mean load [%MBL]'] / 100 * (MBL * 1000 / Area)
df['Smax [MPa]'] = df['Sm [MPa]'] + df['Sa [MPa]']

# Calculate stress ranges for various methods
df['Sr SWT'] = 2 * df['Sa [MPa]'] * Smith_Watson_Topper(df['Sa [MPa]'], df['Smax [MPa]'])
df['Sr Goodman'] = 2 * df['Sa [MPa]'] * Goodman(df['Sm [MPa]'], Su)
df['Sr Gerber'] = 2 * df['Sa [MPa]'] * Gerber(df['Sm [MPa]'], Su)
df['Sr Soderberg'] = 2 * df['Sa [MPa]'] * Soderberg(df['Sm [MPa]'], Sy)
df['Sr Kwofie'] = 2 * df['Sa [MPa]'] * Kwofie(df['Sm [MPa]'], Su, alpha)
df['Sr Morrow'] = 2 * df['Sa [MPa]'] * Morrow(df['Sm [MPa]'], Sfp)

###############################################################################
# Model Fitting
###############################################################################
models = ['Sr SWT', 'Sr Goodman', 'Sr Gerber', 'Sr Soderberg', 'Sr Kwofie', 'Sr Morrow']
model_names = ['SWT', 'Goodman', 'Gerber', 'Soderberg', 'Kwofie', 'Morrow']
fit_results = {}

for model, name in zip(models, model_names):
    x_data = df['Cycles to failure'].values
    y_data = df[model].values

    # Fit to power-law model
    popt, _ = curve_fit(power_law, x_data, y_data)
    y_fit = power_law(x_data, *popt)

    # Calculate R²
    r_squared = calculate_r2(y_data, y_fit)
    fit_results[name] = {'parameters': popt, 'R²': r_squared}

###############################################################################
# Plot Original Data
###############################################################################
plt.figure()
plt.scatter(df.loc[:'A5', 'Cycles to failure'], df.loc[:'A5', 'Load range [MPa]'], label='Sm = 8% MBL', facecolors='none', edgecolors='#006969ff')
plt.scatter(df.loc['A6':, 'Cycles to failure'], df.loc['A6':, 'Load range [MPa]'], label='Sm = 18% MBL', color='#006969ff')

plt.xscale('log')
plt.xlim(1E5, 5E6)
plt.ylim(30, 70)
plt.xlabel('Cycles to failure', fontsize=14)
plt.ylabel('Stress range [MPa]', fontsize=14)
plt.legend(fontsize=14)
plt.show()

###############################################################################
# Plot Normalized Data
###############################################################################
df['Normalized Sr SWT'] = df['Sr SWT'] / np.max(df['Sr SWT'])
df['Normalized Sr Gerber'] = df['Sr Gerber'] / np.max(df['Sr Gerber'])

popt_swt, _ = curve_fit(power_law, df['Cycles to failure'], df['Normalized Sr SWT'])
popt_gerber, _ = curve_fit(power_law, df['Cycles to failure'], df['Normalized Sr Gerber'])

y_fit_swt = power_law(df['Cycles to failure'], *popt_swt)
y_fit_gerber = power_law(df['Cycles to failure'], *popt_gerber)

r2_swt = calculate_r2(df['Normalized Sr SWT'], y_fit_swt)
r2_gerber = calculate_r2(df['Normalized Sr Gerber'], y_fit_gerber)

plt.figure()
plt.scatter(df['Cycles to failure'], df['Normalized Sr SWT'], label='SWT', color='#33ccccff', marker='s')
plt.scatter(df['Cycles to failure'], df['Normalized Sr Gerber'], label='Gerber', color='#006969ff', marker='v')

x_fit = np.linspace(1E5, 5E6, 100)
plt.plot(x_fit, power_law(x_fit, *popt_swt), label=f'SWT fit: $R^2$={r2_swt:.3f}', color='#33ccccff', linestyle='--')
plt.plot(x_fit, power_law(x_fit, *popt_gerber), label=f'Gerber fit: $R^2$={r2_gerber:.3f}', color='#006969ff', linestyle='--')

plt.xscale('log')
plt.xlim(1E5, 5E6)
plt.ylim(0.4, 1.1)
plt.xlabel('Cycles to failure [-]', fontsize=14)
plt.ylabel('Normalised Stress Range [-]', fontsize=14)
plt.legend(fontsize=11, loc='lower left', framealpha=0.8, facecolor='none', edgecolor='black')
plt.show()
