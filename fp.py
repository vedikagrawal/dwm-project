import streamlit as st
import pandas as pd
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# ===================== FP-Growth Code =====================
class FPNode:
    id_counter = 0

    def __init__(self, item_name, count, parent):
        self.item_name = item_name
        self.count = count
        self.parent = parent
        self.children = {}
        self.link = None
        self.uid = FPNode.id_counter
        FPNode.id_counter += 1

    def increment(self, count):
        self.count += count

def build_header_table(transactions, min_support_count):
    item_counts = defaultdict(int)
    for transaction, count in transactions:
        for item in transaction:
            item_counts[item] += count
    return {item: [cnt, None] for item, cnt in item_counts.items() if cnt >= min_support_count}

def sort_items(transaction, header_table):
    return sorted([item for item in transaction if item in header_table],
                  key=lambda item: header_table[item][0], reverse=True)

def insert_tree(transaction, root, header_table, count):
    current_node = root
    for item in transaction:
        if item not in current_node.children:
            new_node = FPNode(item, count, current_node)
            current_node.children[item] = new_node
            if header_table[item][1] is None:
                header_table[item][1] = new_node
            else:
                node = header_table[item][1]
                while node.link:
                    node = node.link
                node.link = new_node
        else:
            current_node.children[item].increment(count)
        current_node = current_node.children[item]

def construct_fp_tree(transactions, min_support_count):
    header_table = build_header_table(transactions, min_support_count)
    if len(header_table) == 0:
        return None, None
    root = FPNode(None, 1, None)
    for transaction, count in transactions:
        sorted_items = sort_items(transaction, header_table)
        if sorted_items:
            insert_tree(sorted_items, root, header_table, count)
    return root, header_table

def ascend_fp_tree(node):
    path = []
    while node.parent is not None and node.parent.item_name is not None:
        node = node.parent
        path.append(node.item_name)
    return path[::-1]

def find_prefix_paths(base_pattern, node):
    conditional_patterns = []
    while node:
        path = ascend_fp_tree(node)
        if path:
            conditional_patterns.append((path, node.count))
        node = node.link
    return conditional_patterns

def mine_fp_tree(header_table, prefix, frequent_itemsets, min_support_count):
    sorted_items = sorted(header_table.items(), key=lambda x: x[1][0])
    for base_item, (count, node) in sorted_items:
        new_freq_set = prefix.copy()
        new_freq_set.add(base_item)
        frequent_itemsets.append((new_freq_set, count))
        conditional_patterns = find_prefix_paths(base_item, node)
        conditional_transactions = []
        for path, cnt in conditional_patterns:
            conditional_transactions.append((path, cnt))
        conditional_tree, new_header_table = construct_fp_tree(conditional_transactions, min_support_count)
        if new_header_table:
            mine_fp_tree(new_header_table, new_freq_set, frequent_itemsets, min_support_count)

def generate_association_rules(frequent_itemsets, min_confidence_percent):
    min_confidence = min_confidence_percent / 100.0  # convert percentage to fraction
    rules = []
    itemset_support = {frozenset(itemset): support for itemset, support in frequent_itemsets}
    for itemset, support in frequent_itemsets:
        if len(itemset) > 1:
            items = list(itemset)
            for i in range(1, len(items)):
                antecedents = combinations(items, i)
                for antecedent in antecedents:
                    antecedent = frozenset(antecedent)
                    consequent = frozenset(itemset) - antecedent
                    if consequent and antecedent in itemset_support:
                        confidence = itemset_support[frozenset(itemset)] / itemset_support[antecedent]
                        if confidence >= min_confidence:
                            rules.append((set(antecedent), set(consequent), round(confidence*100, 2), support))
    return rules

from itertools import combinations

# ===================== Streamlit App =====================
st.set_page_config(page_title="Market Basket Analysis - FP-Growth", layout="wide")
st.title("🛒 Market Basket Analysis using FP-Growth")

st.markdown("""
Upload a **Transaction CSV** and find **Frequent Itemsets** and **Association Rules**  
using the **FP-Growth** algorithm!
""")

uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data")
    st.dataframe(df.head())

    column_options = df.columns.tolist()
    item_col = st.selectbox("Select the column containing items per transaction:", column_options)

    raw_transactions = df[item_col].dropna().astype(str).apply(lambda x: frozenset(x.strip().split(',')))
    transaction_counter = Counter(raw_transactions)

    transactions = [(list(items), count) for items, count in transaction_counter.items()]

    min_support_count = st.number_input("Enter Minimum Support (number of transactions)", min_value=1, value=2)
    min_confidence_percent = st.number_input("Enter Minimum Confidence (%)", min_value=1, max_value=100, value=60)

    if st.button("Run FP-Growth"):
        root, header_table = construct_fp_tree(transactions, min_support_count)

        if root is None:
            st.warning("No frequent items found with the given support.")
        else:
            frequent_itemsets = []
            mine_fp_tree(header_table, set(), frequent_itemsets, min_support_count)

            result_df = pd.DataFrame([(list(items), support) for items, support in frequent_itemsets],
                                     columns=["Itemset", "Support"])

            if result_df.empty:
                st.warning("No frequent itemsets found.")
            else:
                st.success(f"Found {len(result_df)} frequent itemsets!")
                st.write("### Frequent Itemsets")
                st.dataframe(result_df)

                item_count = defaultdict(int)
                for items, support in frequent_itemsets:
                    for item in items:
                        item_count[item] += support

                top_items = sorted(item_count.items(), key=lambda x: x[1], reverse=True)[:10]
                if top_items:
                    item_names = [x[0] for x in top_items]
                    item_support = [x[1] for x in top_items]

                    fig, ax = plt.subplots()
                    ax.bar(item_names, item_support, color='skyblue')
                    plt.xticks(rotation=45)
                    plt.title("Top Frequent Items")
                    st.pyplot(fig)

            rules = generate_association_rules(frequent_itemsets, min_confidence_percent)

            if rules:
                rules_df = pd.DataFrame([(
                    list(antecedent), list(consequent), confidence, support
                ) for antecedent, consequent, confidence, support in rules],
                columns=["Antecedent", "Consequent", "Confidence (%)", "Support"])

                st.write("### Association Rules")
                st.dataframe(rules_df)

                st.download_button("Download Frequent Itemsets", result_df.to_csv(index=False), file_name="frequent_itemsets.csv")
                st.download_button("Download Association Rules", rules_df.to_csv(index=False), file_name="association_rules.csv")
            else:
                st.info("No association rules found with the given confidence.")
