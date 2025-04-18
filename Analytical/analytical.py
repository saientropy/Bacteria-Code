#!/usr/bin/env python3
"""
e_coli_wall_mechanics.py   ―   zero‑touch plotting helper
----------------------------------------------------------------
Recreates the tension‑vs‑strain figures from Albarrán et al. 2023
for both a spherical cap (β=1) and a cylindrical side wall (β≈4).
The code needs *no* user input: just run it and collect the PNGs.

Output
------
    components_<geo>.png   – T_cw, T_pm, σ_total (+ Laplace line)
    psi_<geo>.png          – generalised pre‑strain ψ(ε)
    stiffness_<geo>.png    – trσ / ψ · dg/dε  (structural stiffness)

Dependencies: numpy, matplotlib (pip install matplotlib)
----------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------------------------------
# Immutable physical / material constants (taken directly from the paper)
# --------------------------------------------------------------------------
EPSILON_C = 0.04          # onset of exponential hardening
G0        = 1.0e-2        # N/m, relaxed shear modulus of the wall
K_PM      = 2.5e-1        # N/m, area‑stretch modulus of plasma membrane

# default physiological load used only for the Laplace benchmark
P_TURGOR_ATM = 1.5        # internal pressure in atm
DIAMETER_UM  = 1.0        # cell diameter in µm
ATM2PA       = 101_325    # conversion factor

# geometry presets we want to illustrate
GEOMETRIES = {
    "sphere":    1.0,     # β = L/D
    "cylinder":  4.0,
}

# strain axis (0 … 20 %)
eps = np.linspace(0.0, 0.20, 401)

# --------------------------------------------------------------------------
# Constitutive relations (Albarrán et al., Eqs. 35, 38)
# --------------------------------------------------------------------------
def g(e):            # hardening potential
    return EPSILON_C * (np.exp(e/EPSILON_C) - 1.0)

def dg_de(e):
    return np.exp(e/EPSILON_C)

def psi(e):          # “generalised pre‑strain”
    return g(e)/(1.0 + e) / dg_de(e)

def T_cw(e, beta):
    return G0 * (3*beta - 1)/beta * 2.0 * g(e) / (1.0 + e)

def T_pm(e):
    return 2.0 * K_PM * e

def sigma_total(e, beta):
    return T_cw(e, beta) + T_pm(e)

def sigma_laplace(beta):
    p = P_TURGOR_ATM * ATM2PA
    D = DIAMETER_UM * 1e-6
    return (3*beta - 1)/(8*beta) * p * D      # N/m (independent of ε)

# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------
def savefig(fig, fname):
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    print(f"  →  {fname}")

outdir = Path.cwd()

for label, beta in GEOMETRIES.items():
    # 1) composite tension plot -------------------------------------------
    fig, ax = plt.subplots()
    ax.plot(eps, T_cw(eps, beta),        label=r"$T_{\mathrm{cw}}$")
    ax.plot(eps, T_pm(eps),              label=r"$T_{\mathrm{pm}}$", ls="--")
    ax.plot(eps, sigma_total(eps, beta), label=r"$\sigma = T_{\mathrm{cw}}+T_{\mathrm{pm}}$", lw=2)
    ax.axhline(sigma_laplace(beta),      label=r"$\sigma_{\mathrm{Laplace}}$", ls=":")
    ax.set_xlabel("strain $\\varepsilon$")
    ax.set_ylabel("tension  [N m$^{-1}$]")
    ax.set_title(f"Composite tension (β = {beta:g})")
    ax.legend(frameon=False)
    savefig(fig, outdir / f"components_{label}.png")

    # 2) ψ(ε) --------------------------------------------------------------
    fig2, ax2 = plt.subplots()
    ax2.plot(eps, psi(eps))
    ax2.set_xlabel("strain $\\varepsilon$")
    ax2.set_ylabel("generalised pre‑strain $\\psi$")
    ax2.set_title(f"$\\psi(\\varepsilon)$   (β = {beta:g})")
    savefig(fig2, outdir / f"psi_{label}.png")

    # 3) trσ / ψ · dg/dε  (structural stiffness term) ----------------------
    tr_sigma_over_psi = (3/4) * P_TURGOR_ATM * ATM2PA * DIAMETER_UM*1e-6 / psi(eps) * dg_de(eps)
    fig3, ax3 = plt.subplots()
    ax3.plot(eps, tr_sigma_over_psi)
    ax3.set_xlabel("strain $\\varepsilon$")
    ax3.set_ylabel(r"$\mathrm{tr}\,\sigma / \psi \,\cdot\, dg/d\varepsilon$   [N m$^{-1}$]")
    ax3.set_title(f"Structural stiffness vs strain   (β = {beta:g})")
    savefig(fig3, outdir / f"stiffness_{label}.png")

plt.show()
