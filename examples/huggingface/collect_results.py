import ast
import glob
import os
import re

import matplotlib.pyplot as plt
import pandas as pd


LOG_DIR = "results"  # change if needed


pattern = re.compile(
    r".+_use_liger_(True|False)_batch_size_(\d+)_rep_(\d+)\.log"
)


records = []


for filepath in glob.glob(os.path.join(LOG_DIR, "*.log")):
    filename = os.path.basename(filepath)

    match = pattern.match(filename)

    use_liger = match.group(1) == "True"
    batch_size = int(match.group(2))
    rep = int(match.group(3))

    with open(filepath, "r") as f:
        text = f.read()

    # Extract final dictionary from log
    dict_matches = re.findall(r"\{.*?\}", text, re.DOTALL)

    if not dict_matches:
        print(f"No metrics dict found in {filename}")
        continue

    metrics_str = dict_matches[-1]

    try:
        metrics = ast.literal_eval(metrics_str)

        # Only keep successful runs that completed all 20 iterations
        if int(metrics.get("step", 0)) != 20:
            print(f"Skipping incomplete run: {filename}")
            continue

    except Exception as e:
        print(f"Failed parsing {filename}: {e}")
        continue

    def to_float(x):
        try:
            return float(x)
        except:
            return None

    records.append(
        {
            "use_liger": use_liger,
            "batch_size": batch_size,
            "rep": rep,
            "total_peak_memory_allocated_MB": to_float(
                metrics.get("total_peak_memory_allocated_MB")
            ),
            "avg_tokens_per_second": to_float(
                metrics.get("avg_tokens_per_second")
            ),
        }
    )


# Build dataframe

df = pd.DataFrame(records)

print("\nRaw Data:\n")
print(df.head())


# Average across runs
summary = (
    df.groupby(["use_liger", "batch_size"])
    .agg(
        total_peak_memory_allocated_MB_mean=(
            "total_peak_memory_allocated_MB",
            "mean",
        ),
        total_peak_memory_allocated_MB_std=(
            "total_peak_memory_allocated_MB",
            "std",
        ),
        avg_tokens_per_second_mean=(
            "avg_tokens_per_second",
            "mean",
        ),
        avg_tokens_per_second_std=(
            "avg_tokens_per_second",
            "std",
        ),
        num_runs=("rep", "count"),
    )
    .reset_index()
)


print("\nSummary:\n")
print(summary)


# -----------------------------
# Plot: Memory (Bar Chart)
# -----------------------------

plt.figure(figsize=(10, 6))

batch_sizes = sorted(summary["batch_size"].unique())
x = range(len(batch_sizes))
width = 0.35

no_liger = (
    summary[summary["use_liger"] == False]
    .set_index("batch_size")
    .reindex(batch_sizes)
)

with_liger = (
    summary[summary["use_liger"] == True]
    .set_index("batch_size")
    .reindex(batch_sizes)
)

valid_no_liger = no_liger["total_peak_memory_allocated_MB_mean"].notna()

plt.bar(
    [i - width / 2 for i, valid in zip(x, valid_no_liger) if valid],
    no_liger.loc[
        valid_no_liger,
        "total_peak_memory_allocated_MB_mean",
    ],
    width=width,
    yerr=no_liger.loc[
        valid_no_liger,
        "total_peak_memory_allocated_MB_std",
    ],
    capsize=4,
    label="Liger=False",
    color="gold",
)

plt.bar(
    [i + width / 2 for i in x],
    with_liger["total_peak_memory_allocated_MB_mean"],
    width=width,
    yerr=with_liger["total_peak_memory_allocated_MB_std"],
    capsize=4,
    label="Liger=True",
    color="royalblue",
)

plt.xticks(list(x), batch_sizes)
plt.xlabel("Batch Size")
plt.ylabel("total Peak Memory allocated (MB)")
plt.title("Memory Usage vs Batch Size")
plt.legend()
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig("memory_vs_batch_size.png", dpi=300)


# -----------------------------
# Plot: Throughput (Bar Chart)
# -----------------------------

plt.figure(figsize=(10, 6))

valid_no_liger = no_liger["avg_tokens_per_second_mean"].notna()

plt.bar(
    [i - width / 2 for i, valid in zip(x, valid_no_liger) if valid],
    no_liger.loc[
        valid_no_liger,
        "avg_tokens_per_second_mean",
    ],
    width=width,
    yerr=no_liger.loc[
        valid_no_liger,
        "avg_tokens_per_second_std",
    ],
    capsize=4,
    label="Liger=False",
    color="gold",
)

plt.bar(
    [i + width / 2 for i in x],
    with_liger["avg_tokens_per_second_mean"],
    width=width,
    yerr=with_liger["avg_tokens_per_second_std"],
    capsize=4,
    label="Liger=True",
    color="royalblue",
)

plt.xticks(list(x), batch_sizes)
plt.xlabel("Batch Size")
plt.ylabel("Average Tokens / Second")
plt.title("Throughput vs Batch Size")
plt.legend()
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig("throughput_vs_batch_size.png", dpi=300)


plt.show()


# Optional: save summary table
summary.to_csv("liger_benchmark_summary.csv", index=False)

print("\nSaved:")
print("- memory_vs_batch_size.png")
print("- throughput_vs_batch_size.png")
print("- liger_benchmark_summary.csv")
