"""
FP-Tree (Frequent Pattern Tree) + FP-Growth on a Groceries-style dataset.

- Dataset format: one transaction per line, comma-separated items.
- Example line: "whole milk,bread,eggs"

What this script does:
1) Reads transactions
2) Computes min support count from minsup fraction (default 0.001)
3) Builds an FP-Tree; after EACH transaction insert, prints the current tree
4) Mines frequent itemsets with FP-Growth
5) Generates association rules with min confidence (default 0.8)

Author: you :)
"""

from __future__ import annotations
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from itertools import combinations
import math
import os
from typing import Dict, List, Optional, Tuple, Iterable, Set


# -----------------------------
# Config
# -----------------------------
DATA_PATH = "groceries.csv"   # <-- set to your data file (CSV, one txn per line, comma-separated)
MIN_SUP_FRAC = 0.001          # 0.1% of transactions
MIN_CONF = 0.8
SHOW_TREE_EVOLUTION = True    # prints tree after every transaction (can be noisy for big datasets)
MAX_TREE_PRINT_WIDTH = 120    # truncate long lines a bit for readability


# -----------------------------
# FP-Tree Data Structures
# -----------------------------
@dataclass
class FPNode:
    item: Optional[str]
    count: int
    parent: Optional["FPNode"]
    children: Dict[str, "FPNode"] = field(default_factory=dict)
    node_link: Optional["FPNode"] = None  # next node with the same item

    def increment(self, n: int = 1) -> None:
        self.count += n


class FPTree:
    def __init__(self):
        self.root = FPNode(item=None, count=0, parent=None)
        self.header_table: Dict[str, Tuple[int, Optional[FPNode]]] = {}  # item -> (support_count, first_node)

    def add_transaction(self, items: List[str]) -> None:
        """Insert a single ordered transaction into the FP-tree."""
        current = self.root
        for item in items:
            if item in current.children:
                current.children[item].increment(1)
            else:
                new_node = FPNode(item=item, count=1, parent=current)
                current.children[item] = new_node
                # header table node-link maintenance
                if item not in self.header_table:
                    self.header_table[item] = (0, None)
                # append to node-link chain
                _, first = self.header_table[item]
                if first is None:
                    self.header_table[item] = (0, new_node)
                else:
                    # walk to end of chain
                    last = first
                    while last.node_link is not None:
                        last = last.node_link
                    last.node_link = new_node
            current = current.children[item]

    def build_header_supports(self) -> None:
        """Recompute support counts in header based on node-link chains."""
        for item, (_, first) in list(self.header_table.items()):
            s = 0
            n = first
            while n is not None:
                s += n.count
                n = n.node_link
            self.header_table[item] = (s, self.header_table[item][1])

    def is_single_path(self) -> bool:
        """Check whether tree is a single path (no branching)."""
        node = self.root
        while True:
            if len(node.children) > 1:
                return False
            if len(node.children) == 0:
                break
            node = next(iter(node.children.values()))
        return True

    def pretty_print(self, limit_width: int = 120) -> None:
        """Print the tree in a simple indented text form."""
        def _recurse(node: FPNode, depth: int) -> Iterable[str]:
            for child in node.children.values():
                label = f"{child.item}:{child.count}"
                yield "  " * depth + label
                yield from _recurse(child, depth + 1)

        lines = list(_recurse(self.root, 0))
        if not lines:
            print("(empty tree)")
            return
        for ln in lines:
            if len(ln) > limit_width:
                ln = ln[:limit_width - 3] + "..."
            print(ln)


# -----------------------------
# FP-Growth Mining
# -----------------------------
def ascend_path(node: FPNode) -> List[str]:
    """Return the path of items from a node up to but excluding the root."""
    path = []
    while node.parent is not None and node.parent.item is not None:
        node = node.parent
        path.append(node.item)
    return path


def conditional_pattern_base(tree: FPTree, item: str) -> List[Tuple[List[str], int]]:
    """Collect conditional pattern base for an item: list of (prefix_path, count)."""
    _, first = tree.header_table[item]
    patterns = []
    node = first
    while node is not None:
        prefix = ascend_path(node)
        if prefix:
            patterns.append((prefix, node.count))
        node = node.node_link
    return patterns


def construct_conditional_tree(pattern_base: List[Tuple[List[str], int]], min_support: int) -> Optional[FPTree]:
    """Build a conditional FP-tree from a pattern base."""
    # count items
    item_counts = Counter()
    for path, count in pattern_base:
        for it in path:
            item_counts[it] += count
    # filter infrequent
    freq_items = {it for it, c in item_counts.items() if c >= min_support}
    if not freq_items:
        return None

    # order by frequency desc then lexicographically for determinism
    ordered = sorted(freq_items, key=lambda it: (-item_counts[it], it))

    tree = FPTree()

    # insert filtered, ordered paths
    for path, count in pattern_base:
        filtered = [it for it in path if it in freq_items]
        # order this path by global freq order
        filtered.sort(key=lambda it: (-item_counts[it], it))
        # insert 'count' times (equivalent to increasing node counts)
        current = tree.root
        for it in filtered:
            if it in current.children:
                current.children[it].increment(count)
            else:
                new_node = FPNode(item=it, count=count, parent=current)
                current.children[it] = new_node
                # header maintenance
                if it not in tree.header_table:
                    tree.header_table[it] = (0, None)
                _, first = tree.header_table[it]
                if first is None:
                    tree.header_table[it] = (0, new_node)
                else:
                    last = first
                    while last.node_link is not None:
                        last = last.node_link
                    last.node_link = new_node
            current = current.children[it]

    tree.build_header_supports()
    return tree


def fp_growth(tree: FPTree, items_in_order: List[str], min_support: int) -> Dict[Tuple[str, ...], int]:
    """
    Mine frequent itemsets from an FP-tree.
    Returns dict: itemset (tuple sorted) -> support count.
    """
    tree.build_header_supports()
    freq_itemsets: Dict[Tuple[str, ...], int] = {}

    # items are processed in increasing frequency order for FP-growth
    for item in items_in_order[::-1]:  # reverse since we got them desc
        if item not in tree.header_table:
            continue
        support, _ = tree.header_table[item]
        if support < min_support:
            continue

        # item alone is a frequent itemset
        freq_itemsets[(item,)] = support

        # conditional pattern base & tree
        cpb = conditional_pattern_base(tree, item)
        cond_tree = construct_conditional_tree(cpb, min_support)
        if cond_tree is None:
            continue

        # recursively mine conditional tree
        cond_items_order = sorted(
            [it for it, (cnt, _) in cond_tree.header_table.items() if cnt >= min_support],
            key=lambda it: (-cond_tree.header_table[it][0], it),
        )
        mined = fp_growth(cond_tree, cond_items_order, min_support)

        # add item to each mined pattern to form larger itemsets
        for pattern, cnt in mined.items():
            new_itemset = tuple(sorted(pattern + (item,)))
            freq_itemsets[new_itemset] = cnt

    return freq_itemsets


# -----------------------------
# Utilities: Rules, Printing
# -----------------------------
def generate_rules(freq_itemsets: Dict[Tuple[str, ...], int],
                   min_conf: float,
                   n_transactions: int) -> List[Tuple[Tuple[str, ...], Tuple[str, ...], float, float]]:
    """
    Generate association rules A -> B with confidence >= min_conf.
    Returns list of (A, B, support, confidence).
    """
    # supports map for quick lookup
    supports = {tuple(sorted(k)): v for k, v in freq_itemsets.items()}

    rules = []
    for itemset, sup_ab in supports.items():
        if len(itemset) < 2:
            continue
        items = list(itemset)
        # all non-empty proper subsets as antecedents
        for r in range(1, len(items)):
            for A in combinations(items, r):
                A = tuple(sorted(A))
                B = tuple(sorted(set(items) - set(A)))
                sup_a = supports.get(A)
                if not sup_a:
                    continue
                conf = sup_ab / sup_a
                if conf >= min_conf:
                    sup = sup_ab / n_transactions
                    rules.append((A, B, sup, conf))
    # sort by confidence desc, then support desc
    rules.sort(key=lambda x: (-x[3], -x[2], x[0], x[1]))
    return rules


def print_header_counts(header_table: Dict[str, Tuple[int, Optional[FPNode]]], top: int = 20):
    counts = sorted(((it, cnt) for it, (cnt, _) in header_table.items()), key=lambda x: (-x[1], x[0]))
    print("\nTop frequent items (support counts):")
    for it, c in counts[:top]:
        print(f"  {it}: {c}")


# -----------------------------
# Loading + Preprocessing
# -----------------------------
def read_transactions(path: str) -> List[List[str]]:
    if os.path.exists(path):
        txns: List[List[str]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items = [itm.strip() for itm in line.split(",") if itm.strip()]
                txns.append(items)
        if not txns:
            raise RuntimeError("No transactions found in file.")
        return txns
    else:
        # Fallback tiny sample so script runs even without file
        print(f"WARNING: '{path}' not found. Using a small fallback sample.")
        return [
            ["milk", "bread", "butter"],
            ["bread", "butter"],
            ["milk", "eggs"],
            ["milk", "bread"],
            ["bread", "eggs"],
            ["beer", "diapers", "bread"],
            ["beer", "diapers"],
            ["milk", "diapers", "bread", "butter"],
            ["coffee", "sugar"],
            ["milk", "coffee"],
        ]


def prepare_transactions(txns: List[List[str]], min_support_frac: float) -> Tuple[List[List[str]], Dict[str, int], int]:
    """
    1) Count item supports
    2) Filter infrequent items by minsup fraction
    3) Sort items in each transaction by global frequency desc then lexicographic
    """
    n = len(txns)
    min_support = max(1, math.ceil(min_support_frac * n))

    counts = Counter()
    for t in txns:
        counts.update(set(t))  # count presence per transaction

    # keep only frequent items
    frequent_items = {it for it, c in counts.items() if c >= min_support}
    if not frequent_items:
        raise RuntimeError("No items meet the minimum support. Try lowering MIN_SUP_FRAC.")

    # order key based on frequency desc then lexicographic
    order_key = lambda it: (-counts[it], it)

    clean_txns: List[List[str]] = []
    for t in txns:
        filtered = [it for it in t if it in frequent_items]
        filtered.sort(key=order_key)
        if filtered:
            clean_txns.append(filtered)

    return clean_txns, {it: counts[it] for it in frequent_items}, min_support


# -----------------------------
# Main
# -----------------------------
def main():
    # 1) Load
    txns = read_transactions(DATA_PATH)
    n = len(txns)
    print(f"Loaded {n} transactions.")

    # 2) Prepare
    txns_ordered, item_counts, min_support = prepare_transactions(txns, MIN_SUP_FRAC)
    print(f"Min support fraction = {MIN_SUP_FRAC} -> min count = {min_support}")
    print(f"{len(item_counts)} items meet minsup.")

    # 3) Build FP-Tree (show evolution)
    tree = FPTree()
    # initialize header table keys so order is deterministic
    for it in sorted(item_counts.keys(), key=lambda k: (-item_counts[k], k)):
        tree.header_table[it] = (0, None)

    print("\n=== Building FP-Tree and showing evolution ===")
    for i, t in enumerate(txns_ordered, start=1):
        tree.add_transaction(t)
        if SHOW_TREE_EVOLUTION:
            print(f"\nAfter transaction #{i}: {t}")
            tree.build_header_supports()
            print_header_counts(tree.header_table, top=10)
            tree.pretty_print(limit_width=MAX_TREE_PRINT_WIDTH)

    # final header support refresh
    tree.build_header_supports()

    # 4) Mine frequent itemsets with FP-Growth
    items_in_order = sorted(item_counts, key=lambda it: (-item_counts[it], it))
    freq_itemsets = fp_growth(tree, items_in_order, min_support)
    print(f"\nMined {len(freq_itemsets)} frequent itemsets (minsup count = {min_support}).")

    # show a few itemsets
    sample = sorted(freq_itemsets.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
    print("\nTop frequent itemsets (itemset -> support_count):")
    for itset, cnt in sample:
        print(f"  {itset}: {cnt}")

    # 5) Generate association rules (min confidence)
    rules = generate_rules(freq_itemsets, MIN_CONF, n_transactions=n)
    print(f"\nGenerated {len(rules)} rules with min confidence = {MIN_CONF}.")
    print("\nTop rules (A -> B) [support, confidence]:")
    for A, B, sup, conf in rules[:20]:
        print(f"  {A} -> {B}  [sup={sup:.4f}, conf={conf:.3f}]")


if __name__ == "__main__":
    main()
