# e_coli_wall_plots.py   --   "single‑shot" script
#
# This file generates *every* figure we have discussed so far:
#
#   1) components_<geo>.png       – T_cw, T_pm, σ_total (+ Laplace)  vs ε
#   2) psi_<geo>.png              – ψ(ε)
#   3) stiffness_<geo>.png        – trσ / ψ · dg/dε  vs ε
#   4) rel_excess_area.png        – ΔA/A  vs  p  (both geometries)
#
# where  <geo>  is  sphere  (β = 1)  or  cylinder  (β ≈ 4).
#
# --------------------------------------------------------------------------
# HOW TO USE
# --------------------------------------------------------------------------
# • Copy‑paste the whole file into your working directory.
# • Run “python e_coli_wall_plots.py”.
# • Nine PNGs will appear in the same folder, ready for slides or reports.
#
# No parameters need to be edited, but if you *do* want to explore, search
# for the section “USER‑TWEAKABLE CONSTANTS” below – everything physical
# is concentrated there.
#
# The code is *heavily commented* so that a new reader sees exactly what
# each line implements from Albarrán et al. 2023.
# --------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------------------------------
# USER‑TWEAKABLE CONSTANTS
# --------------------------------------------------------------------------
# Material parameters (from the paper + your slides)
EPSILON_C = 0.04            # onset strain of exponential hardening (ε_c)
G0        = 1.0e-2          # N/m, shear modulus of relaxed cell‑wall mesh
K_PM      = 2.5e-1          # N/m, area‑stretch modulus of the plasma membrane

# Cell geometry & loading for Laplace calculations
DIAMETER_UM  = 1.0          # µm   (1.0 µm ≈ E.coli)
P_TURGOR_ATM = 1.5          # atm  (physiological order of magnitude)
# You can change these two and rerun ‑‑ everything else updates automatically.

# Axis setup
EPS_RANGE   = (0.0, 0.20)   # strain domain displayed (0 … 20 %)
N_PTS       = 401           # resolution of ε‑axis

# Geometries we want to illustrate
GEOMETRIES = {
    "sphere":    1.0,       # β = 1
    "cylinder":  4.0        # β ≈ L/D for a rod‑like side wall
}

# If you need publication‑quality black‑on‑white images, activate the next line
plt.style.use("default")

# --------------------------------------------------------------------------
#   SECTION A – helper functions translating the paper’s formulas
# --------------------------------------------------------------------------
ATM2PA = 101_325                            # pressure conversion factor (Pa / atm)
D_M    = DIAMETER_UM * 1e-6                # cell diameter in metres

def g(eps):
    """Strain‑hardening potential  g(ε)  (Albarrán Eq. 35)."""
    return EPSILON_C * (np.exp(eps / EPSILON_C) - 1.0)

def dg_de(eps):
    """Derivative  dg/dε  – needed for ψ and stiffness."""
    return np.exp(eps / EPSILON_C)

def psi(eps):
    """
    Generalised pre‑strain ψ(ε) = g/(1+ε) · (dg/dε)^−1
    See Eq. (39) in the paper; governs geometric vs intrinsic hardening.
    """
    return g(eps) / (1.0 + eps) / dg_de(eps)

def T_cw(eps, beta):
    """
    Tension carried by the *cell wall* composite mesh.
    Eq. (Composite) on your slide:  T_cw = G0 (3β−1)/β · 2 g(ε)/(1+ε)
    """
    return G0 * (3*beta - 1) / beta * 2.0 * g(eps) / (1.0 + eps)

def T_pm(eps):
    """
    Membrane contribution: small‑strain area expansion of the plasma
    membrane.  Linear law from the slide:  T_pm ≈ 2 K ε
    """
    return 2.0 * K_PM * eps

def sigma_total(eps, beta):
    """Composite surface tension σ = T_cw + T_pm."""
    return T_cw(eps, beta) + T_pm(eps)

def sigma_laplace(beta):
    """
    Baseline Laplace tension that *would* balance pressure if the wall
    were an ideal in‑extensible membrane.
        σ_Lap = (3β−1)/(8β) · p · D
    independent of strain; plotted as a horizontal benchmark.
    """
    p = P_TURGOR_ATM * ATM2PA
    return (3*beta - 1) / (8*beta) * p * D_M

def pressure_from_sigma(eps, beta):
    """
    Invert the Laplace relation to compute the pressure corresponding
    to a *given* composite tension at that ε:
        p = (8β/(3β−1)) σ_total / D
    Returns value in atm so it can be plotted directly.
    """
    sigma = sigma_total(eps, beta)
    p_pa  = 8*beta / (3*beta - 1) * sigma / D_M
    return p_pa / ATM2PA

# --------------------------------------------------------------------------
#   SECTION B – figure generators
# --------------------------------------------------------------------------

def make_component_plot(eps, beta, tag):
    """Plot T_cw, T_pm and σ_total vs ε for one geometry."""
    fig, ax = plt.subplots()
    ax.plot(eps, T_cw(eps, beta),        label=r"$T_{\mathrm{cw}}$")
    ax.plot(eps, T_pm(eps),              label=r"$T_{\mathrm{pm}}$", ls="--")
    ax.plot(eps, sigma_total(eps, beta), label=r"$\sigma_{\mathrm{total}}$", lw=2)
    ax.axhline(sigma_laplace(beta),      label=r"$\sigma_{\mathrm{Laplace}}$", ls=":")
    ax.set_xlabel("strain $\\varepsilon$")
    ax.set_ylabel("tension  [N m$^{-1}$]")
    ax.set_title(f"Composite tension vs strain  (β = {beta:g})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fname = f"components_{tag}.png"
    fig.savefig(fname, dpi=300)
    print(f"  saved {fname}")

def make_psi_plot(eps, tag, beta):
    fig, ax = plt.subplots()
    ax.plot(eps, psi(eps))
    ax.set_xlabel("strain $\\varepsilon$")
    ax.set_ylabel("generalised pre‑strain $\\psi$")
    ax.set_title(f"$\\psi(\\varepsilon)$   (β = {beta:g})")
    fig.tight_layout()
    fname = f"psi_{tag}.png"
    fig.savefig(fname, dpi=300)
    print(f"  saved {fname}")

def make_stiffness_plot(eps, tag, beta):
    """Plot trσ / ψ · dg/dε  (structural part of stiffness matrix)."""
    tr_sigma_over_psi = (3/4) * P_TURGOR_ATM * ATM2PA * D_M / psi(eps) * dg_de(eps)
    fig, ax = plt.subplots()
    ax.plot(eps, tr_sigma_over_psi)
    ax.set_xlabel("strain $\\varepsilon$")
    ax.set_ylabel(r"$\mathrm{tr}\,\sigma / \psi \,\cdot\, dg/d\varepsilon$  [N m$^{-1}$]")
    ax.set_title(f"Structural stiffness vs strain  (β = {beta:g})")
    fig.tight_layout()
    fname = f"stiffness_{tag}.png"
    fig.savefig(fname, dpi=300)
    print(f"  saved {fname}")

def make_rel_excess_plot(eps):
    """
    ΔA/A vs pressure for *both* geometries on the same axes.
    Uses ΔA/A ≈ 2 ε   for small isotropic strains.
    """
    rel_excess = 2 * eps
    fig, ax = plt.subplots()
    for tag, beta in GEOMETRIES.items():
        p = pressure_from_sigma(eps, beta)
        ax.plot(p, rel_excess, label=tag)
    ax.set_xlabel("turgor pressure  $p$  [atm]")
    ax.set_ylabel("relative excess area  $\\Delta A / A$")
    ax.set_title("Relative excess area vs turgor pressure")
    ax.legend(frameon=False)
    fig.tight_layout()
    fname = "rel_excess_area.png"
    fig.savefig(fname, dpi=300)
    print(f"  saved {fname}")

# --------------------------------------------------------------------------
#   SECTION C – run everything
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Create output directory (current directory is fine; mkdir is safe if exists)
    outdir = Path(".")
    outdir.mkdir(exist_ok=True)

    # Strain axis
    eps = np.linspace(*EPS_RANGE, N_PTS)

    # Generate plots per geometry
    for tag, beta in GEOMETRIES.items():
        make_component_plot(eps, beta, tag)
        make_psi_plot(eps, tag, beta)
        make_stiffness_plot(eps, tag, beta)

    # Extra plot combining geometries
    make_rel_excess_plot(eps)

    print("\nAll figures generated – ready to drop into your report/slide deck!")


