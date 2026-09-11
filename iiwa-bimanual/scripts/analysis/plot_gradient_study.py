import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import argparse

def main():
    parser = argparse.ArgumentParser(description="Plot gradient study results.")
    parser.add_argument("--interactive", action="store_true", help="Show plots interactively.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.abspath(os.path.join(script_dir, "../.."))
    data_file = os.path.join(repo_dir, "out/gradient_study_results.csv")
    out_dir = os.path.join(repo_dir, "out/plots/gradient_study")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found. Run the experiment first.")
        return

    df = pd.read_csv(data_file)
    sns.set_theme(style="whitegrid")
    lowess = sm.nonparametric.lowess

    import matplotlib as mpl
    import matplotlib.ticker as ticker
    
    # IEEE Conference Single-Column Format Calibration
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.autolayout': True,
        'pdf.fonttype': 42, # Avoid type 3 fonts for IEEE
        'ps.fonttype': 42
    })

    # 1. Runtime vs Partials Size
    plt.figure(figsize=(3.5, 2.5))
    df["Runtime (us)"] = df["Runtime (s)"] * 1e6
    ax = sns.lineplot(data=df, x="Partials Size", y="Runtime (us)", hue="Method", marker="o", errorbar="sd")
    plt.xscale('log', base=2); plt.yscale('log')
    ax.tick_params(axis='both', which='major', color='black', bottom=True, left=True, direction='out')
    ax.tick_params(axis='both', which='minor', color='lightgray', bottom=True, left=True, direction='out')
    ax.minorticks_on()
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
    
    # Calculate min/max of the means (the points plotted)
    stats = df.groupby(['Method', 'Partials Size'])['Runtime (us)'].mean()
    y_min, y_max = stats.min(), stats.max()
    
    plt.axhline(y_min, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.text(df["Partials Size"].max(), y_min, rf'{y_min:.1f} $\mu s$ ', 
             verticalalignment='bottom', horizontalalignment='right', fontsize=6, color='gray',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.5))
    plt.axhline(y_max, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.text(df["Partials Size"].min(), y_max, rf' {y_max:.1f} $\mu s$', 
             verticalalignment='bottom', fontsize=6, color='gray',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.5))
    
    plt.grid(True, which='major', color='lightgray', alpha=0.5)
    plt.ylabel(r"Runtime [$\mu s$]")
    plt.title('Runtime vs Size', fontsize=10)
    plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.78), frameon=True)
    plt.savefig(os.path.join(out_dir, "runtime_vs_size.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, "runtime_vs_size.png"), bbox_inches='tight', dpi=300)
    
    # 2. Numerical Error vs Partials Size
    plt.figure(figsize=(3.5, 2.5))
    df_err = df[df["Method"] == "IFT AD"].copy()
    df_err["p95"] = df_err.groupby("Partials Size")["Max Error"].transform(lambda x: x.quantile(0.95))
    df_err["Is Outlier"] = df_err["Max Error"] > df_err["p95"]
    sns.lineplot(data=df_err, x="Partials Size", y="Max Error", marker="o", estimator=np.nanmedian, errorbar=("pi", 90), label="Median + 5-95%")
    sns.scatterplot(data=df_err[df_err["Is Outlier"]], x="Partials Size", y="Max Error", color='red', alpha=0.3, s=5, label="Outliers")
    plt.xscale('log', base=2); plt.yscale('log')
    plt.title('Numerical Error vs Size', fontsize=10)
    plt.legend(loc='upper right', frameon=True)
    plt.savefig(os.path.join(out_dir, "error_vs_size.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, "error_vs_size.png"), bbox_inches='tight', dpi=300)

    # 3. Error vs Distance to Boundary
    plt.figure(figsize=(3.5, 2.5))
    df_bound = df_err[df_err["Max Error"] > 0].sort_values("Boundary Distance")
    sns.scatterplot(data=df_bound, x="Boundary Distance", y="Max Error", alpha=0.1, s=2, color='gray')
    z_lowess = lowess(np.log(df_bound["Max Error"]), np.log(df_bound["Boundary Distance"]), frac=0.05)
    plt.plot(np.exp(z_lowess[:, 0]), np.exp(z_lowess[:, 1]), color='red', lw=1, label="LOESS")
    plt.xscale('log'); plt.yscale('log')
    plt.title('Error vs Distance to Boundary', fontsize=10)
    plt.savefig(os.path.join(out_dir, "error_vs_boundary.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, "error_vs_boundary.png"), bbox_inches='tight', dpi=300)

    # 4. Error vs Augmented Condition Number
    plt.figure(figsize=(3.5, 2.5))
    df_aug = df_err[df_err["Max Error"] > 0].sort_values("Cond Augmented")
    sns.scatterplot(data=df_aug, x="Cond Augmented", y="Max Error", alpha=0.1, s=2, color='gray')
    z_aug = lowess(np.log(df_aug["Max Error"]), np.log(df_aug["Cond Augmented"]), frac=0.05)
    plt.plot(np.exp(z_aug[:, 0]), np.exp(z_aug[:, 1]), color='green', lw=1, label="LOESS")
    plt.xscale('log'); plt.yscale('log')
    plt.title('Error vs Condition Number', fontsize=10)
    plt.savefig(os.path.join(out_dir, "error_vs_cond_aug.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, "error_vs_cond_aug.png"), bbox_inches='tight', dpi=300)

    print(f"Plots saved to {out_dir}")

    if args.interactive:
        plt.show()

if __name__ == "__main__":
    main()
