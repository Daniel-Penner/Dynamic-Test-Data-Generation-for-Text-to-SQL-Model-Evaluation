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

RESULTS_DIR = Path("outputs")
PREDS_DIR = RESULTS_DIR / "BIRD"
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

summary = summary.sort_index()
print(summary)

summary.round(4).to_latex(
    LATEX_OUT / "accuracy_runtime_summary.tex",
    caption="Accuracy and runtime comparison for Bird, Synthetic, and Combined evaluation.",
    label="tab:accuracy_runtime",
)


# ============================================================
# STATISTICAL TESTS
# ============================================================

stats = []

for model, g in df.groupby("model"):
    bird = np.array(g["bird_ex"])
    synthetic = np.array(g["synthetic_ex"])
    combined = np.array(g["combined_ex"])

    def mcnemar_stats(a, b):
        n11 = np.sum((a == 1) & (b == 1))
        n10 = np.sum((a == 1) & (b == 0))
        n01 = np.sum((a == 0) & (b == 1))
        n00 = np.sum((a == 0) & (b == 0))

        table = [[n11, n10], [n01, n00]]
        result = mcnemar(table, exact=False, correction=True)

        extra_reject_rate = n10 / (n10 + n01) if (n10 + n01) > 0 else 0.0
        return result.pvalue, extra_reject_rate

    p_syn, extra_syn = mcnemar_stats(bird, synthetic)
    p_comb, extra_comb = mcnemar_stats(bird, combined)

    stats.append({
        "model": model,

        "bird_acc": bird.mean(),
        "synthetic_acc": synthetic.mean(),
        "combined_acc": combined.mean(),

        "risk_diff_bird_vs_synthetic": synthetic.mean() - bird.mean(),
        "risk_diff_bird_vs_combined": combined.mean() - bird.mean(),

        "cohens_h_bird_vs_synthetic": cohens_h(bird.mean(), synthetic.mean()),
        "cohens_h_bird_vs_combined": cohens_h(bird.mean(), combined.mean()),

        "mcnemar_p_bird_vs_synthetic": p_syn,
        "mcnemar_p_bird_vs_combined": p_comb,

        "extra_reject_rate_synthetic": extra_syn,
        "extra_reject_rate_combined": extra_comb,
    })

stats_df = pd.DataFrame(stats).sort_values("model")
print(stats_df)

stats_df.round(4).to_latex(
    LATEX_OUT / "statistical_tests.tex",
    caption="Paired statistical comparison between Bird, Synthetic, and Combined evaluation results.",
    label="tab:stats",
)


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
