import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from common_functions import *

###############################################################################
# Mooring Characteristics
###############################################################################
Nominal_dia = 76  # mm
component = "studless"
grade = "R3"
Su, Sy, MBL = Properties(grade, Nominal_dia).values()
gamma = 0.25
Area = 2 * np.pi * (Nominal_dia / 2) ** 2
mbl_20 = MBL * 1000 / Area * 0.2  # 20% MBL

# S-N Data
k = 3
Sf = 2022  # Morrow

###############################################################################
# Plot Mean Stress Influence
###############################################################################
Sa = np.linspace(5, Su - 50, 120)
Sm = np.linspace(5, Su - 50, 120)
X, Y = np.meshgrid(Sm, Sa)

# Plot settings
plt.rcParams["font.family"] = "Times New Roman"
methods = [
    ("Goodman", lambda Sm, Sa: Goodman(Sm, Su)),
    ("Gerber", lambda Sm, Sa: Gerber(Sm, Su)),
    ("Soderberg", lambda Sm, Sa: Soderberg(Sm, Sy)),
    ("SWT", lambda Sm, Sa: Smith_Watson_Topper(Sa, Sm + Sa)),
    ("Morrow", lambda Sm, Sa: Morrow(Sm, Sf)),
    ("Kwofie", lambda Sm, Sa: Kwofie(Sm, Su, 1)),
]
subplot_labels = ["a)", "b)", "c)", "d)", "e)", "f)"]

fig, axes = plt.subplots(3, 2, figsize=(10, 10))
for ax, (method, func), label in zip(axes.flat, methods, subplot_labels):
    # Compute the relative difference
    Z = (func(X, Y) * Y - Y) / Y * 100
    rel_dif = (func(mbl_20, mbl_20) * mbl_20 - mbl_20) / mbl_20 * 100

    # Contour plot
    pcm = ax.pcolor(X, Y, Z, cmap="viridis", norm=LogNorm(vmin=0.1, vmax=1e3))
    ax.axvline(x=mbl_20, color="white", linestyle="--")
    text_pos = (mbl_20 + 10, 100)

    # Add annotation
    if method == "SWT":
        ax.text(*text_pos, f"20% MBL \n {rel_dif:.1f}% (at Sa = Sm)", color="white")
    else:
        ax.text(*text_pos, f"20% MBL \n {rel_dif:.1f}%", color="white")

    # Titles and labels
    ax.text(mbl_20 + 20, 580, f"{method}", color="white", size=12)
    ax.set_xlabel("Sm [MPa]")
    ax.set_ylabel("Sa [MPa]")
    ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=14, verticalalignment="top", color="white")

    # Colorbar
    fig.colorbar(pcm, ax=ax, label="Relative difference [%]")

plt.tight_layout()
plt.show()
