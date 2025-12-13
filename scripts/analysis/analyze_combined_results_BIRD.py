import json
from pathlib import Path
from typing import List
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar


# ============================================================
# CONFIG
# ============================================================

RESULTS_DIR = Path("outputs/BIRD")
LATEX_OUT = RESULTS_DIR / "latex_tables"
PLOTS_OUT = RESULTS_DIR / "plots"

LATEX_OUT.mkdir(parents=True, exist_ok=True)
PLOTS_OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# STATISTICS
# ============================================================

def cohens_h(p1, p2):
    return 2 * (math.asin(math.sqrt(p2)) - math.asin(math.sqrt(p1)))


# ============================================================
# LOAD ALL RESULTS
# ============================================================

records = []

for f in RESULTS_DIR.glob("*.json"):
    model = f.stem
    if model.endswith("_predictions"):
        model = model.removesuffix("_predictions")

    with open(f, "r", encoding="utf8") as fh:
        data = json.load(fh)

    for q in data["queries"]:
        records.append({
            "model": model,

            "bird_ex": q["bird"]["ex"],
            "synthetic_ex": q["synthetic"]["ex_all_tests"],
            "combined_ex": q["combined"]["ex"],

            "bird_ms": q["runtime"]["bird_ms"],
            "synthetic_ms": q["runtime"]["synthetic_ms"],
            "combined_ms": q["runtime"]["combined_ms"],
        })

df = pd.DataFrame(records)

print(f"[INFO] Loaded {len(df)} per-query records")


# ============================================================
# DERIVED METRICS
# ============================================================

df["bird_reject"] = 1 - df["bird_ex"]
df["synthetic_reject"] = 1 - df["synthetic_ex"]
df["combined_reject"] = 1 - df["combined_ex"]


# ============================================================
# SUMMARY TABLE
# ============================================================

summary = df.groupby("model").agg(
    bird_acc=("bird_ex", "mean"),
    synthetic_acc=("synthetic_ex", "mean"),
    combined_acc=("combined_ex", "mean"),

    bird_time=("bird_ms", "mean"),
    synthetic_time=("synthetic_ms", "mean"),
    combined_time=("combined_ms", "mean"),
)

table_df = pd.DataFrame({
    "Model": summary.index,
    "EX(A)": (summary["bird_acc"] * 100).round(2),
    "EX(B)": (summary["synthetic_acc"] * 100).round(2),
    "EX(C)": (summary["combined_acc"] * 100).round(2),

    "Time(A)": summary["bird_time"].round(2).astype(str) + "ms",
    "Time(B)": summary["synthetic_time"].round(2).astype(str) + "ms",
    "Time(C)": summary["combined_time"].round(2).astype(str) + "ms",
})

latex_str = r"""
\begin{table}
\small
\resizebox{\columnwidth}{!}{
\begin{tabular}{lrrrrrr}
\toprule
\textbf{Model} & \textbf{EX(A)} & \textbf{EX(B)} & \textbf{EX(C)} &
\textbf{Time(A)} & \textbf{Time(B)} & \textbf{Time(C)} \\
\midrule
"""

for _, row in table_df.iterrows():
    latex_str += (
        f"{row['Model']} & "
        f"{row['EX(A)']:.2f} & {row['EX(B)']:.2f} & {row['EX(C)']:.2f} & "
        f"{row['Time(A)']} & {row['Time(B)']} & {row['Time(C)']} \\\\\n"
    )

latex_str += r"""\bottomrule
\end{tabular}
}
\caption{Accuracy and runtime comparison on Spider dataset (A), synthetic tests (B), and combined evaluation (C).}
\label{tab:accuracy_runtime_spider}
\end{table}
"""

with open(LATEX_OUT / "accuracy_runtime_pretty.tex", "w", encoding="utf8") as f:
    f.write(latex_str)

print("[INFO] Pretty Spider LaTeX table written!")


# ============================================================
# STATISTICAL TESTS — BIRD (A vs C) WITH Δ AND P-VALUE ONLY
# ============================================================

from statsmodels.stats.contingency_tables import mcnemar

stats_rows = []

for model, g in df.groupby("model"):
    bird = np.array(g["bird_ex"])
    combined = np.array(g["combined_ex"])

    # 2×2 contingency table
    n11 = np.sum((bird == 1) & (combined == 1))
    n10 = np.sum((bird == 1) & (combined == 0))
    n01 = np.sum((bird == 0) & (combined == 1))
    n00 = np.sum((bird == 0) & (combined == 0))

    table = [[n11, n10], [n01, n00]]
    p_val = mcnemar(table, exact=False, correction=True).pvalue

    # Δ accuracy (percentage points)
    delta = (combined.mean() - bird.mean()) * 100

    stats_rows.append({
        "Model": model,
        "EX(A)": bird.mean() * 100,
        "EX(B)": g["synthetic_ex"].mean() * 100,
        "EX(C)": combined.mean() * 100,
        "Δ(A→C)": delta,
        "p(A vs C)": p_val
    })

stats_df = pd.DataFrame(stats_rows).sort_values("Model")

# Format columns
stats_df["EX(A)"] = stats_df["EX(A)"].round(2)
stats_df["EX(B)"] = stats_df["EX(B)"].round(2)
stats_df["EX(C)"] = stats_df["EX(C)"].round(2)
stats_df["Δ(A→C)"] = stats_df["Δ(A→C)"].round(2)
stats_df["p(A vs C)"] = stats_df["p(A vs C)"].apply(lambda p: f"{p:.2e}")

# Write LaTeX
latex = []
latex.append("\\begin{table}")
latex.append("\\small")
latex.append("\\resizebox{\\columnwidth}{!}{")
latex.append("\\begin{tabular}{lrrrrr}")
latex.append("\\toprule")
latex.append("\\textbf{Model} & \\textbf{EX(A)} & \\textbf{EX(B)} & \\textbf{EX(C)} & "
             "\\textbf{$\\Delta$(A→C)} & \\textbf{$p$-value} \\\\")
latex.append("\\midrule")

for _, row in stats_df.iterrows():
    latex.append(
        f"{row['Model']} & "
        f"{row['EX(A)']:.2f} & "
        f"{row['EX(B)']:.2f} & "
        f"{row['EX(C)']:.2f} & "
        f"{row['Δ(A→C)']:.2f} & "
        f"{row['p(A vs C)']} \\\\"
    )

latex.append("\\bottomrule")
latex.append("\\end{tabular}")
latex.append("}")
latex.append("\\caption{Statistical comparison for BIRD: Δ accuracy (A→C) and McNemar $p$-value.}")
latex.append("\\label{tab:bird_stats}")
latex.append("\\end{table}")

(LATEX_OUT / "bird_stat_tests.tex").write_text("\n".join(latex), encoding="utf8")
print("[INFO] Wrote BIRD statistical tests table → bird_stat_tests.tex")


# ============================================================
# BOXPLOT — REJECTED QUERIES
# ============================================================

reject_rates = df.groupby("model").agg(
    bird_reject=("bird_reject", "mean"),
    synthetic_reject=("synthetic_reject", "mean"),
    combined_reject=("combined_reject", "mean"),
).reset_index()

df_reject_long = df.melt(
    id_vars="model",
    value_vars=["bird_reject", "synthetic_reject", "combined_reject"],
    var_name="eval",
    value_name="rejected"
)

df_reject_long["eval"] = df_reject_long["eval"].map({
    "bird_reject": "Bird",
    "synthetic_reject": "Synthetic",
    "combined_reject": "Combined",
})


plt.figure(figsize=(9, 5))

x = np.arange(len(reject_rates))
width = 0.25

plt.scatter(x - width, reject_rates["bird_reject"], label="Bird", s=60)
plt.scatter(x, reject_rates["synthetic_reject"], label="Synthetic", s=60)
plt.scatter(x + width, reject_rates["combined_reject"], label="Combined", s=60)

plt.xticks(x, reject_rates["model"], rotation=45, ha="right")
plt.ylabel("Rejection rate (1 − EX)")
plt.title("Rejection rate per model and evaluation type")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig(PLOTS_OUT / "rejection_rate_per_model.png", dpi=300)
plt.close()

# ============================================================
# BOXPLOT — REJECTION RATE ACROSS MODELS (CORRECT)
# ============================================================

# Compute model-level rejection rates
reject_rates = df.groupby("model").agg(
    Bird=("bird_reject", "mean"),
    Synthetic=("synthetic_reject", "mean"),
    Combined=("combined_reject", "mean"),
)

# Prepare data for boxplot: 9 values per dataset
data = [
    reject_rates["Bird"].values,
    reject_rates["Synthetic"].values,
    reject_rates["Combined"].values,
]

plt.figure(figsize=(7, 5))

plt.boxplot(
    data,
    labels=["Bird", "Synthetic", "Combined"],
    showfliers=True,
)

# Overlay individual model points
for i, vals in enumerate(data, start=1):
    x = np.random.normal(i, 0.04, size=len(vals))  # jitter
    plt.scatter(x, vals, alpha=0.8, s=40)

plt.ylabel("Rejection rate (1 − EX)")
plt.title("Model-level rejection rates across evaluation types")

plt.tight_layout()
plt.savefig(PLOTS_OUT / "rejection_rate_boxplot_across_models.png", dpi=300)
plt.close()



reject_rates = df.groupby("model").agg(
    bird=("bird_reject", "mean"),
    synthetic=("synthetic_reject", "mean"),
    combined=("combined_reject", "mean"),
)


x = np.arange(len(reject_rates))
width = 0.18  # thinner bars

plt.figure(figsize=(10, 4))

plt.bar(x - width, reject_rates["bird"], width, label="Bird")
plt.bar(x, reject_rates["synthetic"], width, label="Synthetic")
plt.bar(x + width, reject_rates["combined"], width, label="Combined")

plt.xticks(x, reject_rates.index, rotation=45, ha="right")
plt.ylabel("Rejection rate (1 − EX)")
plt.title("Average rejection rate per model")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig(PLOTS_OUT / "rejection_rate_barplot.png", dpi=300)
plt.close()


# ============================================================
# BOXPLOT — RUNTIME PER QUERY
# ============================================================

avg_time = df.groupby("model").agg(
    bird=("bird_ms", "mean"),
    synthetic=("synthetic_ms", "mean"),
    combined=("combined_ms", "mean"),
)

x = np.arange(len(avg_time))
width = 0.18   # thinner bars → more spacing

plt.figure(figsize=(10.5, 4))

plt.bar(x - width, avg_time["bird"], width, label="Bird")
plt.bar(x, avg_time["synthetic"], width, label="Synthetic")
plt.bar(x + width, avg_time["combined"], width, label="Combined")

plt.xticks(x, avg_time.index, rotation=45, ha="right")
plt.ylabel("Time per query (ms)")
plt.title("Average runtime per model")

# create visual space for legend
plt.legend(loc="upper left")

plt.tight_layout()
plt.savefig(PLOTS_OUT / "runtime_barplot_spaced.png", dpi=300)
plt.close()




# ============================================================
# BAR PLOT — RUNTIME BEFORE / AFTER
# ============================================================

avg_time = df.groupby("model").agg(
    bird=("bird_ms", "mean"),
    synthetic=("synthetic_ms", "mean"),
    combined=("combined_ms", "mean"),
)

x = np.arange(len(avg_time))
width = 0.30

plt.figure(figsize=(10, 5))

plt.bar(x - width, avg_time["bird"], width, label="Bird")
plt.bar(x, avg_time["synthetic"], width, label="Synthetic")
plt.bar(x + width, avg_time["combined"], width, label="Combined")

plt.xticks(x, avg_time.index, rotation=45, ha="right")
plt.ylabel("Time per query (ms)")
plt.title("Runtime comparison per model")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
plt.tight_layout()
plt.savefig(PLOTS_OUT / "runtime_barplot.png", dpi=300)
plt.close()


print("[INFO] Analysis complete")
print(f"[INFO] LaTeX tables written to: {LATEX_OUT}")
print(f"[INFO] Plots written to: {PLOTS_OUT}")
