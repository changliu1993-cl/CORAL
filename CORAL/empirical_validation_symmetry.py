import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def load_summary_and_records(path):
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "records" not in data:
        raise ValueError("Expect top-level dict with a 'records' field.")
    records = [r for r in data["records"] if isinstance(r, dict)]
    if len(records) == 0:
        raise ValueError("records is empty.")
    print(f"[Load] top-level keys: {list(data.keys())}")
    print(f"[Load] #records: {len(records)}")
    print(f"[Load] example record keys: {list(records[0].keys())}")
    return data, records


def filter_by_answer(records, answer_value):
    # answer_value: "no" for negative POPE, "yes" for positive POPE
    out = []
    for r in records:
        a = str(r.get("answer", "")).strip().lower()
        if a == answer_value:
            out.append(r)
    return out


def compute_deltas(records, mode="margin_diff", key_gm="GM_stat_yes"):
    """
    mode:
      - "margin_diff": Δ = (logit_pos_yes - logit_pos_no) - (logit_neg_yes - logit_neg_no)
      - "gm": Δ = record[key_gm]
    """
    deltas = []
    skipped = 0

    if mode == "margin_diff":
        required = ["logit_pos_yes", "logit_pos_no", "logit_neg_yes", "logit_neg_no"]
        for r in records:
            if not all(k in r for k in required):
                skipped += 1
                continue
            try:
                pos_yes = float(r["logit_pos_yes"])
                pos_no  = float(r["logit_pos_no"])
                neg_yes = float(r["logit_neg_yes"])
                neg_no  = float(r["logit_neg_no"])
                delta = (pos_yes - pos_no) - (neg_yes - neg_no)
                deltas.append(delta)
            except Exception:
                skipped += 1

    elif mode == "gm":
        for r in records:
            if key_gm not in r:
                skipped += 1
                continue
            try:
                deltas.append(float(r[key_gm]))
            except Exception:
                skipped += 1
    else:
        raise ValueError("Unknown mode. Use 'margin_diff' or 'gm'.")

    return np.array(deltas, dtype=float), skipped


# -----------------------------
# Plotting: Histogram + QQ
# -----------------------------
def plot_mirror_symmetry(deltas, save_prefix, bins=80):
    if deltas.size < 10:
        raise ValueError("Too few deltas to assess symmetry.")

    neg_deltas = -deltas

    # ---------- Histogram ----------
    plt.figure(figsize=(10, 8))
    plt.hist(
        deltas, bins=bins, density=True,
        alpha=0.6, label=r"$\Delta_i$", color="#4C72B0"
    )
    plt.hist(
        neg_deltas, bins=bins, density=True,
        alpha=0.6, label=r"$-\Delta_i$", color="#DD8452"
    )
    plt.axvline(0, linestyle="--", linewidth=2, color="black", alpha=0.8)

    plt.xlabel(r"$\Delta_i$", fontsize=28)
    plt.ylabel("Density", fontsize=28)
    plt.legend(fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_hist.png", dpi=300)
    plt.show()

    # ---------- QQ plot ----------
    plt.figure(figsize=(10, 8))
    qs = np.linspace(0.01, 0.99, 99)
    q1 = np.quantile(deltas, qs)
    q2 = np.quantile(neg_deltas, qs)

    plt.scatter(q1, q2, s=30, alpha=0.7)
    lo, hi = min(q1.min(), q2.min()), max(q1.max(), q2.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2)

    plt.xlabel(r"Quantiles of $\Delta_i$", fontsize=28)
    plt.ylabel(r"Quantiles of $-\Delta_i$", fontsize=28)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.gca().set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_qq.png", dpi=300)
    plt.show()

    # ---------- Numeric diagnostics ----------
    mean = deltas.mean()
    skew = stats.skew(deltas)
    ks = stats.ks_2samp(deltas, neg_deltas)
    corr = np.corrcoef(deltas, neg_deltas)[0, 1]

    print(f"N = {deltas.size}")
    print(f"mean(Δ) = {mean:.6f}")
    print(f"|mean(Δ)| = {abs(mean):.6f}")
    print(f"skew(Δ) = {skew:.4f}")
    print(f"corr(Δ, -Δ) = {corr:.3f}")
    print(f"KS statistic = {ks.statistic:.4f}, p = {ks.pvalue:.4g}")

def main():
    path = "pope_random_llava_gm.json" 
    save_tag = "pope_random_llava"

    summary, records = load_summary_and_records(path)

    # ---- Choose Δ definition ----
    delta_mode = "margin_diff"


    # ---- Split into negative/positive by ground-truth answer ----
    neg_records = filter_by_answer(records, "no")    # negative POPE: guaranteed absent
    pos_records = filter_by_answer(records, "yes")

    print(f"[Split] negative(no): {len(neg_records)} | positive(yes): {len(pos_records)}")

    # ---- Compute Δ on negative set ----
    deltas_neg, skipped_neg = compute_deltas(neg_records, mode=delta_mode)
    print(f"[Delta-neg] computed={deltas_neg.size}, skipped={skipped_neg}")

    plot_mirror_symmetry(deltas_neg, save_prefix=f"{save_tag}_{delta_mode}_NEG")


if __name__ == "__main__":
    main()
