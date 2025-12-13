import json
from pathlib import Path
import pandas as pd
import numpy as np
import math

RESULTS_DIR = Path("outputs/Spider")
LATEX_OUT = RESULTS_DIR / "latex_tables"
LATEX_OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD RESULTS
# ============================================================

records = []

for f in RESULTS_DIR.glob("*.json"):
    model = f.stem.replace("_predictions", "")

    with open(f, "r", encoding="utf8") as fh:
        data = json.load(fh)

    for q in data["queries"]:
        records.append({
            "model": model,

            # Spider EX accuracy
            "spider_ex": q["spider"]["ex"],
            "synthetic_ex": q["synthetic"]["ex_all_tests"],
            "combined_ex": q["combined"]["ex"],

            # Runtime
            "spider_ms": q["runtime"]["spider_ms"],
            "synthetic_ms": q["runtime"]["synthetic_ms"],
            "combined_ms": q["runtime"]["combined_ms"],
        })

df = pd.DataFrame(records)
print(f"[INFO] Loaded {len(df)} per-query records")


# ============================================================
# SUMMARY
# ============================================================

summary = df.groupby("model").agg(
    spider_acc=("spider_ex", "mean"),
    synthetic_acc=("synthetic_ex", "mean"),
    combined_acc=("combined_ex", "mean"),

    spider_time=("spider_ms", "mean"),
    synthetic_time=("synthetic_ms", "mean"),
    combined_time=("combined_ms", "mean"),
).sort_index()

# --- Do NOT convert accuracy to string ---
summary["spider_acc"] = summary["spider_acc"] * 100
summary["synthetic_acc"] = summary["synthetic_acc"] * 100
summary["combined_acc"] = summary["combined_acc"] * 100

# Round to 2 decimals (still numeric)
summary["spider_acc"] = summary["spider_acc"].round(2)
summary["synthetic_acc"] = summary["synthetic_acc"].round(2)
summary["combined_acc"] = summary["combined_acc"].round(2)

# --- Times CAN be strings because they never get numeric formatting ---
summary["spider_time"] = summary["spider_time"].round(2).astype(str) + "ms"
summary["synthetic_time"] = summary["synthetic_time"].round(2).astype(str) + "ms"
summary["combined_time"] = summary["combined_time"].round(2).astype(str) + "ms"


# ============================================================
# STATISTICAL TESTS FOR SPIDER
# ============================================================

from statsmodels.stats.contingency_tables import mcnemar

def cohens_h(p1, p2):
    return 2 * (math.asin(math.sqrt(p2)) - math.asin(math.sqrt(p1)))

def mcnemar_stats(a, b):
    n11 = np.sum((a == 1) & (b == 1))
    n10 = np.sum((a == 1) & (b == 0))
    n01 = np.sum((a == 0) & (b == 1))
    n00 = np.sum((a == 0) & (b == 0))

    table = [[n11, n10], [n01, n00]]
    result = mcnemar(table, exact=False, correction=True)

    return result.pvalue


stats_rows = []

for model, g in df.groupby("model"):
    spider = np.array(g["spider_ex"])
    synthetic = np.array(g["synthetic_ex"])
    combined = np.array(g["combined_ex"])

    # p-values
    p_syn = mcnemar_stats(spider, synthetic)
    p_comb = mcnemar_stats(spider, combined)

    # Cohen's h
    h_syn = cohens_h(spider.mean(), synthetic.mean())
    h_comb = cohens_h(spider.mean(), combined.mean())

    stats_rows.append({
        "Model": model,

        "EX(A)": round(spider.mean() * 100, 2),
        "EX(B)": round(synthetic.mean() * 100, 2),
        "EX(C)": round(combined.mean() * 100, 2),

        "p(A vs B)": p_syn,
        "h(A vs B)": h_syn,

        "p(A vs C)": p_comb,
        "h(A vs C)": h_comb,
    })

stats_df = pd.DataFrame(stats_rows).sort_values("Model")


# Convert p-values → scientific notation string
stats_df["p(A vs B)"] = stats_df["p(A vs B)"].apply(lambda p: f"{p:.2e}")
stats_df["p(A vs C)"] = stats_df["p(A vs C)"].apply(lambda p: f"{p:.2e}")

# Round Cohen's h
stats_df["h(A vs B)"] = stats_df["h(A vs B)"].round(4)
stats_df["h(A vs C)"] = stats_df["h(A vs C)"].round(4)


# ============================================================
# STATISTICAL TESTS (SPIDER A vs COMBINED C)
# ============================================================

from statsmodels.stats.contingency_tables import mcnemar
import math

def cohens_h(p1, p2):
    return 2 * (math.asin(math.sqrt(p2)) - math.asin(math.sqrt(p1)))

stats_rows = []

for model, g in df.groupby("model"):
    spider = np.array(g["spider_ex"])
    combined = np.array(g["combined_ex"])

    # Build 2×2 contingency table
    n11 = np.sum((spider == 1) & (combined == 1))
    n10 = np.sum((spider == 1) & (combined == 0))
    n01 = np.sum((spider == 0) & (combined == 1))
    n00 = np.sum((spider == 0) & (combined == 0))

    table = [[n11, n10], [n01, n00]]
    result = mcnemar(table, exact=False, correction=True)

    p_val = result.pvalue
    h_val = cohens_h(spider.mean(), combined.mean())

    stats_rows.append({
        "Model": model,
        "EX(A)": spider.mean() * 100,
        "EX(C)": combined.mean() * 100,
        "p(A vs C)": p_val,
        "h(A vs C)": h_val,
    })

stats_df = pd.DataFrame(stats_rows).sort_values("Model")

# Round EX columns
stats_df["EX(A)"] = stats_df["EX(A)"].round(2)
stats_df["EX(C)"] = stats_df["EX(C)"].round(2)

# Format p-values in scientific notation
stats_df["p(A vs C)"] = stats_df["p(A vs C)"].apply(lambda p: f"{p:.2e}")

# Round Cohen's h
stats_df["h(A vs C)"] = stats_df["h(A vs C)"].round(4)


# ============================================================
# WRITE LATEX TABLE — Spider Statistical Tests
# ============================================================

# ============================================================
# STATISTICAL TESTS — DELTAS + P-VALUES ONLY
# ============================================================

from statsmodels.stats.contingency_tables import mcnemar

def mcnemar_p(a, b):
    """Return McNemar p-value between two binary EX vectors."""
    n11 = np.sum((a == 1) & (b == 1))
    n10 = np.sum((a == 1) & (b == 0))
    n01 = np.sum((a == 0) & (b == 1))
    n00 = np.sum((a == 0) & (b == 0))
    table = [[n11, n10], [n01, n00]]
    return mcnemar(table, exact=False, correction=True).pvalue


stats_rows = []

for model, g in df.groupby("model"):
    spider = np.array(g["spider_ex"])
    synthetic = np.array(g["synthetic_ex"])
    combined = np.array(g["combined_ex"])

    exA = spider.mean() * 100
    exB = synthetic.mean() * 100
    exC = combined.mean() * 100

    # deltas
    dAB = exB - exA
    dAC = exC - exA

    # McNemar p-values
    pAB = mcnemar_p(spider, synthetic)
    pAC = mcnemar_p(spider, combined)

    stats_rows.append({
        "Model": model,
        "EX(A)": round(exA, 2),
        "EX(B)": round(exB, 2),
        "EX(C)": round(exC, 2),

        "Δ(A,B)": round(dAB, 2),
        "Δ(A,C)": round(dAC, 2),

        "p(A,B)": f"{pAB:.2e}",
        "p(A,C)": f"{pAC:.2e}",
    })

stats_df = pd.DataFrame(stats_rows).sort_values("Model")


# ============================================================
# WRITE LATEX TABLE — DELTAS + P-VALUES ONLY
# ============================================================

latex = []
latex.append("\\begin{table}")
latex.append("\\small")
latex.append("\\centering")
latex.append("\\resizebox{\\columnwidth}{!}{")
latex.append("\\begin{tabular}{lrrrrrrr}")
latex.append("\\toprule")
latex.append("\\textbf{Model} & "
             "\\textbf{EX(A)} & \\textbf{EX(B)} & \\textbf{EX(C)} & "
             "\\textbf{$\\Delta$(A,B)} & \\textbf{$\\Delta$(A,C)} & "
             "\\textbf{$p$(A,B)} & \\textbf{$p$(A,C)} \\\\")
latex.append("\\midrule")

for _, row in stats_df.iterrows():
    latex.append(
        f"{row['Model']} & "
        f"{row['EX(A)']:.2f} & "
        f"{row['EX(B)']:.2f} & "
        f"{row['EX(C)']:.2f} & "
        f"{row['Δ(A,B)']:.2f} & "
        f"{row['Δ(A,C)']:.2f} & "
        f"{row['p(A,B)']} & "
        f"{row['p(A,C)']} \\\\"
    )

latex.append("\\bottomrule")
latex.append("\\end{tabular}")
latex.append("}")
latex.append("\\caption{Statistical comparison of Spider EX(A), Synthetic EX(B), Combined EX(C), accuracy deltas, and McNemar significance tests.}")
latex.append("\\label{tab:spider_stats_deltas}")
latex.append("\\end{table}")

(LATEX_OUT / "spider_stat_tests_deltas.tex").write_text("\n".join(latex), encoding="utf8")

print("[INFO] Wrote Spider Δ + p-value statistical table → spider_stat_tests_deltas.tex")


# ============================================================
# BUILD LATEX TABLE (EXACT STYLE REQUESTED)
# ============================================================

latex_lines = []
latex_lines.append("\\begin{table}")
latex_lines.append("\\small")
latex_lines.append("\\resizebox{\\columnwidth}{!}{")
latex_lines.append("\\begin{tabular}{lrrrrrr}")
latex_lines.append("\\toprule")
latex_lines.append("\\textbf{Model} & "
                   "\\textbf{EX(A)} & \\textbf{EX(B)} & \\textbf{EX(C)} & "
                   "\\textbf{Time(A)} & \\textbf{Time(B)} & \\textbf{Time(C)} \\\\")
latex_lines.append("\\midrule")

for model, row in summary.iterrows():
    latex_lines.append(
        f"{model} & "
        f"{row['spider_acc']:.2f} & "
        f"{row['synthetic_acc']:.2f} & "
        f"{row['combined_acc']:.2f} & "
        f"{row['spider_time']} & "
        f"{row['synthetic_time']} & "
        f"{row['combined_time']} \\\\"
    )

latex_lines.append("\\bottomrule")
latex_lines.append("\\end{tabular}")
latex_lines.append("}")
latex_lines.append("\\caption{Accuracy and average runtime per query comparison on Spider dataset (A), TestQL synthetic tests (B), and Combined evaluation (C).}")
latex_lines.append("\\label{tab:spider_accuracy_runtime}")
latex_lines.append("\\end{table}")

out_file = LATEX_OUT / "accuracy_runtime_summary_SPIDER.tex"
out_file.write_text("\n".join(latex_lines), encoding="utf8")

print(f"[INFO] Wrote table → {out_file}")
