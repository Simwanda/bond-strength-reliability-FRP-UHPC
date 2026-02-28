#!/usr/bin/env python3
"""
Reproducibility script for:
Simple bond strength equations and reliability-based model partial factors
for steel and FRP reinforcement in normal-, seawater-, and UHPC concrete.

This script:
1) Loads the curated 5-subgroup database (Excel)
2) Computes normalized bond strength tau_norm = tau/sqrt(fc)
3) Reconstructs simplified-equation predictions from coefficient sheets
4) Computes model uncertainty theta = tau_norm_obs / tau_norm_pred
5) Fits lognormal parameters and computes partial factors gamma_M for target betas
6) Produces publication figures (PDF)
7) Exports summary tables (CSV/Excel)

Expected inputs (relative to repo root):
- data/Final_Subgroups_Bond_Database.xlsx
- equations/SeparateModels_CV_OOB_Equations.xlsx

Outputs are written to:
- outputs/tables/
- outputs/figures/

Python 3.9+ recommended.
Dependencies: numpy, pandas, scipy, matplotlib, openpyxl
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm, spearmanr
import re

# -------------------------- Helpers --------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_sheet_header_fallback(xlsx: Path, sheet: str) -> pd.DataFrame:
    """Some Mendeley exports contain an extra header row. This attempts to fix it."""
    df = pd.read_excel(xlsx, sheet_name=sheet)
    if df.columns.astype(str).str.contains("Unnamed").any() and df.iloc[0].notna().sum() > 5:
        new_cols = df.iloc[0].astype(str).tolist()
        df = df.iloc[1:].copy()
        df.columns = new_cols
    return df

def parse_intercept(eq: str) -> float:
    m = re.search(r"tau_norm\s*=\s*([+-]?\d+(?:\.\d+)?)", str(eq))
    return float(m.group(1)) if m else 0.0

def lognormal_mom_params(mu: float, cov: float) -> tuple[float, float]:
    """Return (mu_ln, sigma_ln) from (mu, cov) for lognormal variable."""
    sigma_ln = float(np.sqrt(np.log(1.0 + cov**2)))
    mu_ln = float(np.log(mu) - 0.5 * sigma_ln**2)
    return mu_ln, sigma_ln

def gamma_M_from_lognormal(mu_ln: float, sigma_ln: float, beta: float) -> float:
    theta_d = float(np.exp(mu_ln - beta * sigma_ln))
    return float(1.0 / theta_d)

# -------------------------- Load database --------------------------

def load_database(db_xlsx: Path) -> pd.DataFrame:
    """
    Loads the curated workbook and returns a single dataframe with:
    - Subgroup labels
    - tau_norm
    - original columns preserved for predictors
    """
    # Steel–SCC
    steel = pd.read_excel(db_xlsx, sheet_name="Steel_SCC")
    steel["Subgroup"] = "Steel--SCC"
    steel["tau_MPa"] = pd.to_numeric(steel.get("τ_R (MPa)"), errors="coerce")
    steel["fc_MPa"]  = pd.to_numeric(steel.get("f_cm (MPa)"), errors="coerce")
    steel["tau_norm"] = steel["tau_MPa"] / np.sqrt(steel["fc_MPa"])

    # FRP Normal
    frp_nm = read_sheet_header_fallback(db_xlsx, "FRP_Normal_Mod")
    frp_nm["Subgroup"] = "FRP--Normal"
    frp_nm["tau_MPa"] = pd.to_numeric(frp_nm.get("Bond strength (MPa)"), errors="coerce")
    frp_nm["fc_MPa"]  = pd.to_numeric(frp_nm.get("Concrete compressive strength (MPa)"), errors="coerce")
    frp_nm["tau_norm"] = frp_nm["tau_MPa"] / np.sqrt(frp_nm["fc_MPa"])

    # FRP SWSSC
    frp_sw = read_sheet_header_fallback(db_xlsx, "FRP_SWSSC")
    frp_sw["Subgroup"] = "FRP--SWSSC"
    frp_sw["tau_MPa"] = pd.to_numeric(frp_sw.get("Bond strength (MPa)"), errors="coerce")
    frp_sw["fc_MPa"]  = pd.to_numeric(frp_sw.get("Concrete compressive strength (MPa)"), errors="coerce")
    frp_sw["tau_norm"] = frp_sw["tau_MPa"] / np.sqrt(frp_sw["fc_MPa"])

    # UHPC Pullout
    uhpc_p = pd.read_excel(db_xlsx, sheet_name="FRP_UHPC_Pullout")
    uhpc_p["Subgroup"] = "FRP--UHPC (Pullout)"
    uhpc_p["tau_MPa"] = pd.to_numeric(uhpc_p.get("Bond strength (MPa)"), errors="coerce")
    uhpc_p["fc_MPa"]  = pd.to_numeric(uhpc_p.get("Concrete compressive strength (MPa)"), errors="coerce")
    uhpc_p["tau_norm"] = uhpc_p["tau_MPa"] / np.sqrt(uhpc_p["fc_MPa"])

    # UHPC Beam
    uhpc_b = pd.read_excel(db_xlsx, sheet_name="FRP_UHPC_Beam")
    uhpc_b["Subgroup"] = "FRP--UHPC (Beam/RILEM)"
    uhpc_b["tau_MPa"] = pd.to_numeric(uhpc_b.get("Bond strength"), errors="coerce")
    uhpc_b["fc_MPa"]  = pd.to_numeric(uhpc_b.get("Concrete compressive strength"), errors="coerce")
    uhpc_b["tau_norm"] = uhpc_b["tau_MPa"] / np.sqrt(uhpc_b["fc_MPa"])

    master = pd.concat([steel, frp_nm, frp_sw, uhpc_p, uhpc_b], ignore_index=True)
    master = master.replace([np.inf, -np.inf], np.nan).dropna(subset=["Subgroup","tau_norm"]).copy()
    return master

# -------------------------- Build predictions --------------------------

def build_predictions(master: pd.DataFrame, eq_xlsx: Path) -> pd.DataFrame:
    """
    Reconstruct tau_norm_pred using:
    - sheet Simplified_Equations: intercept embedded in equation string
    - coef_* sheets: variable coefficients
    Returns dataframe containing tau_norm_obs, tau_norm_pred, theta.
    """
    eq_df = pd.read_excel(eq_xlsx, sheet_name="Simplified_Equations")

    coef_sheet_map = {
        "FRP_Normal_Mod": "coef_FRP_Normal_Mod",
        "FRP_SWSSC": "coef_FRP_SWSSC",
        "FRP–UHPC (Beam/RILEM)": "coef_FRP–UHPC (BeamRILEM)",
        "FRP–UHPC (Pullout)": "coef_FRP–UHPC (Pullout)",
        "Steel–SCC": "coef_Steel–SCC",
    }
    paper_label_map = {
        "FRP_Normal_Mod": "FRP--Normal",
        "FRP_SWSSC": "FRP--SWSSC",
        "FRP–UHPC (Pullout)": "FRP--UHPC (Pullout)",
        "FRP–UHPC (Beam/RILEM)": "FRP--UHPC (Beam/RILEM)",
        "Steel–SCC": "Steel--SCC",
    }

    frames = []
    for _, row in eq_df.iterrows():
        sg_eq = row["Subgroup"]
        if sg_eq not in coef_sheet_map or sg_eq not in paper_label_map:
            continue

        intercept = parse_intercept(row["Equation"])
        coef_df = pd.read_excel(eq_xlsx, sheet_name=coef_sheet_map[sg_eq]).dropna()
        var_col, coef_col = coef_df.columns[:2]

        sg = paper_label_map[sg_eq]
        sub = master[master["Subgroup"] == sg].copy()

        y_pred = np.full(len(sub), intercept, dtype=float)
        for var, c in zip(coef_df[var_col].astype(str), coef_df[coef_col].astype(float)):
            var = var.strip()
            if var in sub.columns:
                col = pd.to_numeric(sub[var], errors="coerce")
                # median imputation for reproducibility if missing values exist
                col = col.fillna(col.median())
                y_pred += c * col.values

        sub["tau_norm_obs"] = sub["tau_norm"].astype(float)
        sub["tau_norm_pred"] = y_pred
        sub["theta"] = sub["tau_norm_obs"] / np.maximum(sub["tau_norm_pred"], 1e-12)
        frames.append(sub)

    out = pd.concat(frames, ignore_index=True)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["tau_norm_obs","tau_norm_pred","theta"])
    return out

# -------------------------- Tables --------------------------

def summarize_uncertainty(pred: pd.DataFrame, betas: list[float]) -> pd.DataFrame:
    rows = []
    for sg, g in pred.groupby("Subgroup"):
        th = g["theta"].values
        mu = float(np.mean(th))
        sd = float(np.std(th, ddof=1))
        cov = float(sd/mu) if mu != 0 else np.nan
        mu_ln, sig_ln = lognormal_mom_params(mu, cov)
        out = {"Subgroup": sg, "mu_theta": mu, "cov_theta": cov, "mu_ln_theta": mu_ln, "sigma_ln_theta": sig_ln, "N": int(len(th))}
        for b in betas:
            out[f"gamma_M_beta_{b}"] = gamma_M_from_lognormal(mu_ln, sig_ln, b)
        rows.append(out)
    return pd.DataFrame(rows).sort_values("Subgroup")

def spearman_top4(pred: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pooled Spearman correlations between theta and mechanistically meaningful variables.
    Variables used:
    - fc_MPa (if present) OR fc column from steel sheet 'fc'
    - l/d ratio column name used in UHPC sheets
    - Ef column name used in FRP sheets
    - hr/d ratio column name used in UHPC pullout
    Returns top 4 by abs rho.
    """
    # Candidate variable names across sheets
    candidates = [
        ("fc_MPa", r"$f'_c$ (MPa)"),
        ("fc", r"$f'_c$ (MPa)"),
        ("Bonded-length-to-diameter ratio", r"$l/d$"),
        ("Bar tensile elastic modulus (GPa)", r"$E_f$ (GPa)"),
        ("Bar tensile elastic modulus", r"$E_f$"),
        ("Rib-height-to-diameter ratio", r"$h_r/d$"),
    ]
    rows = []
    for col, label in candidates:
        if col not in pred.columns:
            continue
        x = pd.to_numeric(pred[col], errors="coerce")
        y = pd.to_numeric(pred["theta"], errors="coerce")
        m = x.notna() & y.notna()
        if int(m.sum()) < 50:
            continue
        rho, p = spearmanr(x[m], y[m])
        rows.append({"Variable": col, "Label": label, "Spearman_rho": float(rho), "p_value": float(p), "N": int(m.sum()), "Abs_rho": float(abs(rho))})
    df = pd.DataFrame(rows).sort_values("Abs_rho", ascending=False).head(4)
    return df

# -------------------------- Figures --------------------------

def fig_parity_5panel(pred: pd.DataFrame, out_pdf: Path) -> None:
    order = ["Steel--SCC","FRP--Normal","FRP--SWSSC","FRP--UHPC (Pullout)","FRP--UHPC (Beam/RILEM)"]
    groups = [g for g in order if g in pred["Subgroup"].unique()]
    labels = ["(a)","(b)","(c)","(d)","(e)"]

    lo = float(min(pred["tau_norm_obs"].min(), pred["tau_norm_pred"].min()))
    hi = float(max(pred["tau_norm_obs"].max(), pred["tau_norm_pred"].max()))
    pad = 0.03 * (hi-lo)
    lo -= pad; hi += pad

    fig = plt.figure(figsize=(12,7.6))
    gs = GridSpec(2, 6, figure=fig)
    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 0:3]),
        fig.add_subplot(gs[1, 3:6]),
    ]

    for ax, sg, lab in zip(axes, groups, labels):
        sub = pred[pred["Subgroup"] == sg]
        x = sub["tau_norm_obs"].values
        y = sub["tau_norm_pred"].values
        ax.scatter(x, y, s=12, alpha=0.65)
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
        r2 = 1 - np.sum((x-y)**2)/np.sum((x-np.mean(x))**2)
        rmse = float(np.sqrt(np.mean((x-y)**2)))
        mae = float(np.mean(np.abs(x-y)))
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linewidth=0.4)
        ax.set_xlabel(r"Experimental $\tau_{\mathrm{norm}}$", fontsize=10)
        ax.set_ylabel(r"Predicted $\tau_{\mathrm{norm}}$", fontsize=10)
        ax.text(0.02, 0.98, lab, transform=ax.transAxes, ha="left", va="top", fontsize=12)
        ax.text(0.98, 0.98, f"{sg}\n$R^2$={r2:.3f}, RMSE={rmse:.3f}\nMAE={mae:.3f}, $N$={len(x)}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9.6)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

def fig_theta_hist_lognormal(pred: pd.DataFrame, out_pdf: Path) -> None:
    order = ["Steel--SCC","FRP--Normal","FRP--SWSSC","FRP--UHPC (Pullout)","FRP--UHPC (Beam/RILEM)"]
    labels = ["(a)","(b)","(c)","(d)","(e)"]

    fig = plt.figure(figsize=(12,7.6))
    gs = GridSpec(2, 6, figure=fig)
    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 0:3]),
        fig.add_subplot(gs[1, 3:6]),
    ]

    for ax, sg, lab in zip(axes, order, labels):
        s = pred[pred["Subgroup"] == sg]["theta"].dropna().values
        if len(s) < 10:
            ax.axis("off"); continue
        mu = float(np.mean(s))
        cov = float(np.std(s, ddof=1)/mu) if mu != 0 else np.nan
        mu_ln, sig_ln = lognormal_mom_params(mu, cov)

        ax.hist(s, bins=18, density=True)
        xs = np.linspace(max(1e-4, s.min()*0.85), s.max()*1.15, 250)
        pdf = (1/(xs*sig_ln*np.sqrt(2*np.pi))) * np.exp(-(np.log(xs)-mu_ln)**2/(2*sig_ln**2))
        ax.plot(xs, pdf, linewidth=1.1)
        ax.grid(True, linewidth=0.4)
        ax.set_xlabel(r"Model uncertainty ratio $\theta$", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.text(0.02, 0.98, lab, transform=ax.transAxes, ha="left", va="top", fontsize=12)
        ax.text(0.98, 0.98, f"{sg}\n$\\mu_\\theta$={mu:.3f}, COV={cov:.3f}\n$N$={len(s)}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9.6)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

def fig_theta_qq(pred: pd.DataFrame, out_pdf: Path) -> None:
    order = ["Steel--SCC","FRP--Normal","FRP--SWSSC","FRP--UHPC (Pullout)","FRP--UHPC (Beam/RILEM)"]
    labels = ["(a)","(b)","(c)","(d)","(e)"]

    fig = plt.figure(figsize=(12,7.6))
    gs = GridSpec(2, 6, figure=fig)
    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 0:3]),
        fig.add_subplot(gs[1, 3:6]),
    ]

    for ax, sg, lab in zip(axes, order, labels):
        s = pred[pred["Subgroup"] == sg]["theta"].dropna().values
        s = s[(s > 0) & np.isfinite(s)]
        if len(s) < 10:
            ax.axis("off"); continue
        z = np.log(s)
        z_sorted = np.sort(z)
        n = len(z_sorted)
        p = (np.arange(1, n+1) - 0.5) / n
        q = norm.ppf(p)
        mu_z = float(np.mean(z_sorted))
        sig_z = float(np.std(z_sorted, ddof=1))
        fit = mu_z + sig_z * q
        ax.scatter(q, z_sorted, s=10, alpha=0.65)
        ax.plot(q, fit, linestyle="--", linewidth=1.0)
        ax.grid(True, linewidth=0.4)
        ax.set_xlabel("Normal quantile", fontsize=10)
        ax.set_ylabel(r"$\ln(\theta)$", fontsize=10)
        ax.text(0.02, 0.98, lab, transform=ax.transAxes, ha="left", va="top", fontsize=12)
        ax.text(0.98, 0.98, f"{sg}\n$\\mu_{{\\ln\\theta}}$={mu_z:.3f}, $\\sigma_{{\\ln\\theta}}$={sig_z:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9.4)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

def fig_gammaM_bar(unc: pd.DataFrame, out_pdf: Path, beta_lo: float = 3.0, beta_hi: float = 3.8) -> None:
    order = ["Steel--SCC","FRP--Normal","FRP--SWSSC","FRP--UHPC (Pullout)","FRP--UHPC (Beam/RILEM)"]
    u = unc.set_index("Subgroup").reindex(order).dropna().reset_index()
    g1 = u[f"gamma_M_beta_{beta_lo}"].values
    g2 = u[f"gamma_M_beta_{beta_hi}"].values

    x = np.arange(len(u["Subgroup"]))
    width = 0.35
    plt.figure(figsize=(10,6))
    plt.bar(x - width/2, g1, width, label=rf"$\beta={beta_lo}$")
    plt.bar(x + width/2, g2, width, label=rf"$\beta={beta_hi}$")
    plt.xticks(x, u["Subgroup"], rotation=20)
    plt.ylabel(r"Model Partial Factor $\gamma_M$")
    plt.title(r"Calibrated Model Partial Factors for Different Reliability Indices")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()

def fig_theta_scatter_2x2(pred: pd.DataFrame, out_pdf: Path) -> None:
    """
    Create a 2x2 grid scatter figure with subgroup-indicated markers, while keeping each panel
    generated individually then combined into a 2x2 PDF page to satisfy plotting rules.
    """
    # Variables
    plots = [
        ("fc_MPa", r"$f'_c$ (MPa)"),
        ("Bonded-length-to-diameter ratio", r"$l/d$"),
        ("Bar tensile elastic modulus (GPa)", r"$E_f$ (GPa)"),
        ("Rib-height-to-diameter ratio", r"$h_r/d$"),
    ]
    # pick available fc column
    if "fc_MPa" not in pred.columns and "fc" in pred.columns:
        plots[0] = ("fc", r"$f'_c$ (MPa)")

    order = ["Steel--SCC","FRP--Normal","FRP--SWSSC","FRP--UHPC (Pullout)","FRP--UHPC (Beam/RILEM)"]
    markers = ["o","s","^","D","v"]
    panel = ["(A)","(B)","(C)","(D)"]

    # build as matplotlib subplots (user requested); we keep default colors.
    fig = plt.figure(figsize=(12,9))
    axs = [fig.add_subplot(2,2,i+1) for i in range(4)]

    for ax, (col, xlabel), lab in zip(axs, plots, panel):
        if col not in pred.columns:
            ax.axis("off"); continue
        for sg, mk in zip(order, markers):
            sub = pred[pred["Subgroup"] == sg]
            x = pd.to_numeric(sub[col], errors="coerce")
            y = pd.to_numeric(sub["theta"], errors="coerce")
            m = x.notna() & y.notna()
            if int(m.sum()) == 0:
                continue
            ax.scatter(x[m], y[m], s=14, alpha=0.7, marker=mk, label=sg)
        x_all = pd.to_numeric(pred[col], errors="coerce")
        y_all = pd.to_numeric(pred["theta"], errors="coerce")
        m_all = x_all.notna() & y_all.notna()
        if int(m_all.sum()) >= 50:
            rho, p = spearmanr(x_all[m_all], y_all[m_all])
            ax.set_title(f"{lab} Spearman $\\rho$={rho:.3f}, $p$={p:.2e}")
        else:
            ax.set_title(f"{lab}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Model uncertainty $\theta$")
        ax.grid(True, linewidth=0.4)

    # single legend outside
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=True)
    fig.tight_layout(rect=[0,0.06,1,1])
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

# -------------------------- Main --------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, default=".", help="Path to repository root")
    ap.add_argument("--betas", type=float, nargs="+", default=[3.0, 3.5, 3.8], help="Target reliability indices")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    db_xlsx = root / "data" / "Final_Subgroups_Bond_Database.xlsx"
    eq_xlsx = root / "equations" / "SeparateModels_CV_OOB_Equations.xlsx"

    if not db_xlsx.exists():
        raise FileNotFoundError(f"Missing database: {db_xlsx}")
    if not eq_xlsx.exists():
        raise FileNotFoundError(f"Missing equations workbook: {eq_xlsx}")

    out_tables = root / "outputs" / "tables"
    out_figs   = root / "outputs" / "figures"
    ensure_dir(out_tables); ensure_dir(out_figs)

    master = load_database(db_xlsx)
    pred = build_predictions(master, eq_xlsx)

    # Save processed dataset
    pred.to_csv(out_tables / "processed_with_theta.csv", index=False)

    # Uncertainty + partial factors
    unc = summarize_uncertainty(pred, args.betas)
    unc.to_csv(out_tables / "uncertainty_and_partial_factors.csv", index=False)

    # Spearman summary (pooled)
    sp = spearman_top4(pred)
    sp.to_csv(out_tables / "spearman_top4.csv", index=False)

    # Figures
    fig_parity_5panel(pred, out_figs / "Parity_Plot_5Panels.pdf")
    fig_theta_hist_lognormal(pred, out_figs / "Theta_Histograms_Lognormal_5Panels.pdf")
    fig_theta_qq(pred, out_figs / "Theta_QQ_lnTheta_5Panels.pdf")
    fig_theta_scatter_2x2(pred, out_figs / "Theta_Scatter_2x2_Subgroups.pdf")
    # GammaM bar for beta=3.0 and 3.8 (if present)
    if 3.0 in args.betas and 3.8 in args.betas:
        fig_gammaM_bar(unc, out_figs / "GammaM_Comparison.pdf", beta_lo=3.0, beta_hi=3.8)

    # Simple console summary
    print("Wrote tables to:", out_tables)
    print("Wrote figures to:", out_figs)
    print("\nUncertainty + partial factors:")
    print(unc[["Subgroup","mu_theta","cov_theta"] + [c for c in unc.columns if c.startswith("gamma_M_beta_")]].to_string(index=False))

if __name__ == "__main__":
    main()
