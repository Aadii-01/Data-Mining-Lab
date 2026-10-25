"""
Experiment 9: Attribute Selection for Classification
Computes Information Gain and Gini Index for categorical and numerical
attributes and selects the best attribute for splitting.
Only basic Python constructs are used (no statistical libraries);
math.log2 is the only function imported.
"""
from math import log2

# ---------------- Dataset (14 records) ----------------
HEADER = ["Outlook", "Windy", "Temperature", "Humidity", "Play"]
CATEGORICAL = ["Outlook", "Windy"]
NUMERICAL = ["Temperature", "Humidity"]
DATA = [
    ["Sunny",    "No",  85, 85, "No"],
    ["Sunny",    "Yes", 80, 90, "No"],
    ["Overcast", "No",  83, 78, "Yes"],
    ["Rainy",    "No",  70, 96, "Yes"],
    ["Rainy",    "No",  68, 80, "Yes"],
    ["Rainy",    "Yes", 65, 70, "No"],
    ["Overcast", "Yes", 64, 65, "Yes"],
    ["Sunny",    "No",  72, 95, "No"],
    ["Sunny",    "No",  69, 70, "Yes"],
    ["Rainy",    "No",  75, 80, "Yes"],
    ["Sunny",    "Yes", 75, 70, "Yes"],
    ["Overcast", "Yes", 72, 90, "Yes"],
    ["Overcast", "No",  81, 75, "Yes"],
    ["Rainy",    "Yes", 71, 80, "No"],
]
TARGET = len(HEADER) - 1


print("Name: Aaditya Shirke \n PRN:23610074 \n Subject: Data Mining Lab")

def class_counts(rows):
    counts = {}
    for r in rows:
        counts[r[TARGET]] = counts.get(r[TARGET], 0) + 1
    return counts


def entropy(rows):
    n = len(rows)
    if n == 0:
        return 0.0
    e = 0.0
    for c in class_counts(rows).values():
        p = c / n
        e -= p * log2(p)
    return e


def gini(rows):
    n = len(rows)
    if n == 0:
        return 0.0
    g = 1.0
    for c in class_counts(rows).values():
        g -= (c / n) ** 2
    return g


def categorical_split(rows, col):
    groups = {}
    for r in rows:
        groups.setdefault(r[col], []).append(r)
    return groups


def evaluate_partition(rows, parts):
    """Return (information gain, weighted gini, gini gain) for a split."""
    n = len(rows)
    w_entropy = sum(len(p) / n * entropy(p) for p in parts)
    w_gini = sum(len(p) / n * gini(p) for p in parts)
    return entropy(rows) - w_entropy, w_gini, gini(rows) - w_gini


def best_numeric_split(rows, col):
    """Try every midpoint between consecutive distinct sorted values."""
    values = []
    for r in rows:
        if r[col] not in values:
            values.append(r[col])
    # simple selection sort (no sorted() / statistics functions)
    for i in range(len(values)):
        m = i
        for j in range(i + 1, len(values)):
            if values[j] < values[m]:
                m = j
        values[i], values[m] = values[m], values[i]
    best = None
    tried = []
    for i in range(len(values) - 1):
        t = (values[i] + values[i + 1]) / 2
        left = [r for r in rows if r[col] <= t]
        right = [r for r in rows if r[col] > t]
        ig, wg, gg = evaluate_partition(rows, [left, right])
        tried.append((t, ig, wg))
        if best is None or ig > best[1]:
            best = (t, ig, wg, gg)
    return best, tried


def main():
    print("DATASET (14 records)")
    print("%-3s" % "#", "  ".join("%-11s" % h for h in HEADER))
    for i, r in enumerate(DATA, 1):
        print("%-3d" % i, "  ".join("%-11s" % v for v in r))

    print("\nTARGET CLASS DISTRIBUTION:", class_counts(DATA))
    print("Entropy of dataset   = %.4f" % entropy(DATA))
    print("Gini Index of dataset = %.4f" % gini(DATA))

    results = []  # (attribute, type, split description, info gain, gini index)

    print("\n=== CATEGORICAL ATTRIBUTES ===")
    for name in CATEGORICAL:
        col = HEADER.index(name)
        groups = categorical_split(DATA, col)
        print("\nAttribute:", name)
        for val, rows in groups.items():
            print("  %-9s n=%-2d classes=%-20s entropy=%.4f gini=%.4f"
                  % (val, len(rows), class_counts(rows), entropy(rows), gini(rows)))
        ig, wg, _ = evaluate_partition(DATA, list(groups.values()))
        print("  Information Gain = %.4f | Weighted Gini = %.4f" % (ig, wg))
        results.append((name, "Categorical", "multiway split", ig, wg))

    print("\n=== NUMERICAL ATTRIBUTES (best binary split point) ===")
    for name in NUMERICAL:
        col = HEADER.index(name)
        best, tried = best_numeric_split(DATA, col)
        print("\nAttribute:", name)
        print("  %-12s %-10s %-10s" % ("Split <=", "InfoGain", "WtdGini"))
        for t, ig, wg in tried:
            print("  %-12.1f %-10.4f %-10.4f" % (t, ig, wg))
        print("  Best split: %s <= %.1f | Information Gain = %.4f | Weighted Gini = %.4f"
              % (name, best[0], best[1], best[2]))
        results.append((name, "Numerical", "%s <= %.1f" % (name, best[0]), best[1], best[2]))

    print("\n=== COMPARISON OF RESULTS ===")
    print("%-12s %-12s %-20s %-10s %-10s" % ("Attribute", "Type", "Split", "InfoGain", "Gini"))
    for a, t, s, ig, g in results:
        print("%-12s %-12s %-20s %-10.4f %-10.4f" % (a, t, s, ig, g))

    best_ig = results[0]
    best_gi = results[0]
    for r in results:
        if r[3] > best_ig[3]:
            best_ig = r
        if r[4] < best_gi[4]:
            best_gi = r
    print("\nBest attribute by Information Gain (highest): %s (%.4f)" % (best_ig[0], best_ig[3]))
    print("Best attribute by Gini Index (lowest)        : %s (%.4f)" % (best_gi[0], best_gi[4]))
    if best_ig[0] == best_gi[0]:
        print("Both criteria agree -> SPLITTING ATTRIBUTE: %s" % best_ig[0])
    else:
        print("Criteria disagree; Information Gain choice used -> %s" % best_ig[0])


if __name__ == "__main__":
    main()
