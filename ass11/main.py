import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# -----------------------------
# Load Dataset
# -----------------------------
dataset_path = "retail.csv"  # replace with your dataset
transactions = []

try:
    with open(dataset_path, "r") as file:
        for line in file:
            transactions.append(line.strip().split(","))
except FileNotFoundError:
    print("Dataset not found! Using sample data.")
    transactions = [
        ["milk", "bread", "butter"],
        ["bread", "jam"],
        ["milk", "bread", "biscuits"],
        ["tea", "sugar", "milk"],
        ["bread", "butter"],
        ["milk", "sugar", "biscuits"]
    ]

# -----------------------------
# Encoding Transactions
# -----------------------------
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# -----------------------------
# FP-Growth Algorithm
# -----------------------------
min_support = 0.2  # adjust based on data
frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)

print("\n✅ Frequent Itemsets:")
print(frequent_itemsets)

# -----------------------------
# Association Rules
# -----------------------------
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# Sorting rules by confidence
rules = rules.sort_values(by="confidence", ascending=False)

print("\n✅ Association Rules (sorted by confidence):")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# -----------------------------
# Top 5 Insights
# -----------------------------
print("\n🔥 Top 5 Strong Rules (for business insights):")
top_rules = rules.head(5)
print(top_rules[['antecedents', 'consequents', 'confidence', 'lift']])
