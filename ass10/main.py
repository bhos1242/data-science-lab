# apriori_groceries.py
# Apriori on grocery dataset with minsup=0.001 and minconf=0.8
# Prints top 5 rules sorted by confidence and marks strong ones.

import os
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# ---------------------------
# Config
# ---------------------------
DATA_PATH = "groceries.csv"    # one transaction per line, comma-separated
MIN_SUP = 0.001                # 0.1% support
MIN_CONF = 0.8                 # 80% confidence
LIFT_STRONG = 1.0              # >1 means positively associated. Try 1.2 for stricter.

# ---------------------------
# Load transactions
# ---------------------------
def load_transactions(path: str):
    if not os.path.exists(path):
        # Fallback tiny sample so the script runs even without a file
        print(f"WARNING: '{path}' not found. Using a small fallback sample.")
        return [
            ["whole milk", "bread", "butter"],
            ["whole milk", "yogurt"],
            ["bread", "butter"],
            ["beer", "diapers", "bread"],
            ["beer", "diapers"],
            ["whole milk", "bread", "yogurt"],
            ["coffee", "sugar"],
            ["whole milk", "coffee"],
            ["eggs", "bread"],
            ["whole milk", "eggs", "butter"],
        ]
    txns = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            txns.append([x.strip() for x in line.split(",") if x.strip()])
    return txns

# ---------------------------
# Main
# ---------------------------
def main():
    txns = load_transactions(DATA_PATH)
    n_txn = len(txns)
    print(f"Loaded {n_txn} transactions.")

    # One-hot encode transactions
    te = TransactionEncoder()
    te_ary = te.fit(txns).transform(txns, sparse=True)  # sparse for big data
    df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)

    # Frequent itemsets via Apriori
    freq = apriori(df, min_support=MIN_SUP, use_colnames=True)
    freq.sort_values(["support", "itemsets"], ascending=[False, True], inplace=True)
    print(f"Found {len(freq)} frequent itemsets with support ≥ {MIN_SUP}.")

    # Association rules (filter by confidence)
    rules = association_rules(freq, metric="confidence", min_threshold=MIN_CONF)

    # Build a nice string representation of the rule
    def set_to_str(s):
        return ", ".join(sorted(list(s)))

    rules["rule"] = rules.apply(
        lambda r: f"{set_to_str(r['antecedents'])}  →  {set_to_str(r['consequents'])}",
        axis=1
    )

    # Mark strong rules: conf ≥ MIN_CONF and lift > LIFT_STRONG
    rules["Strong"] = (rules["confidence"] >= MIN_CONF) & (rules["lift"] > LIFT_STRONG)

    # Sort and select top 5 by confidence, then lift, then support
    # Check which columns exist in the dataframe
    available_cols = ["rule", "support", "confidence", "lift"]
    optional_cols = ["antecedent support", "consequent support", "leverage", "conviction", "zhangs_metric"]
    
    for col in optional_cols:
        if col in rules.columns:
            available_cols.append(col)
    available_cols.append("Strong")
    
    rules_sorted = rules.sort_values(
        by=["confidence", "lift", "support"], ascending=[False, False, False]
    )[available_cols].head(5)

    # Pretty print
    pd.set_option("display.max_colwidth", 120)
    print("\nTop 5 association rules (sorted by confidence):")
    print(rules_sorted.to_string(index=False))

    # Optional: save to CSV
    rules_sorted.to_csv("top5_rules.csv", index=False)
    print("\nSaved: top5_rules.csv")

    # Visualization
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get top 10 rules for better visualization
    top_rules = rules.sort_values(
        by=["confidence", "lift", "support"], ascending=[False, False, False]
    ).head(10)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Association Rules Analysis', fontsize=16, fontweight='bold')
    
    # 1. Support vs Confidence scatter plot
    ax1 = axes[0, 0]
    scatter = ax1.scatter(top_rules['support'], top_rules['confidence'], 
                          c=top_rules['lift'], s=200, alpha=0.6, cmap='viridis', 
                          edgecolors='black', linewidth=1.5)
    ax1.set_xlabel('Support', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Confidence', fontsize=12, fontweight='bold')
    ax1.set_title('Support vs Confidence (colored by Lift)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Lift')
    
    # 2. Lift values bar chart
    ax2 = axes[0, 1]
    rule_labels = [f"Rule {i+1}" for i in range(len(top_rules))]
    colors = ['#2ecc71' if strong else '#e74c3c' for strong in top_rules['Strong']]
    bars = ax2.barh(rule_labels, top_rules['lift'].values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Lift', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Rules', fontsize=12, fontweight='bold')
    ax2.set_title('Lift Values (Green=Strong, Red=Weak)', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()
    
    # 3. Confidence bar chart
    ax3 = axes[1, 0]
    ax3.barh(rule_labels, top_rules['confidence'].values, color='#3498db', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Confidence', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Rules', fontsize=12, fontweight='bold')
    ax3.set_title('Confidence Values', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.invert_yaxis()
    ax3.set_xlim([0, 1.05])
    
    # 4. Support bar chart
    ax4 = axes[1, 1]
    ax4.barh(rule_labels, top_rules['support'].values, color='#e67e22', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Support', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Rules', fontsize=12, fontweight='bold')
    ax4.set_title('Support Values', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('association_rules_visualization.png', dpi=300, bbox_inches='tight')
    print("\nSaved: association_rules_visualization.png")
    plt.show()
    
    # Additional visualization: Network-style plot
    fig2, ax = plt.subplots(figsize=(14, 10))
    
    # Create a scatter plot with annotations
    for idx, row in top_rules.head(10).iterrows():
        # Plot point
        x = row['support']
        y = row['confidence']
        size = row['lift'] * 20
        color = '#2ecc71' if row['Strong'] else '#e74c3c'
        
        ax.scatter(x, y, s=size, alpha=0.6, c=color, edgecolors='black', linewidth=2)
        
        # Add rule text
        rule_text = row['rule'].replace(' →', '\n→')
        if len(rule_text) > 50:
            rule_text = rule_text[:47] + '...'
        ax.annotate(rule_text, (x, y), fontsize=8, ha='center', va='center', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax.set_xlabel('Support', fontsize=14, fontweight='bold')
    ax.set_ylabel('Confidence', fontsize=14, fontweight='bold')
    ax.set_title('Association Rules Visualization\n(Size represents Lift, Color: Green=Strong, Red=Weak)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, top_rules['support'].max() * 1.1])
    ax.set_ylim([0.75, 1.05])
    
    plt.tight_layout()
    plt.savefig('association_rules_network.png', dpi=300, bbox_inches='tight')
    print("Saved: association_rules_network.png")
    plt.show()

if __name__ == "__main__":
    main()
