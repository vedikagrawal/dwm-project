import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations
import math

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

class FPTree:
    def __init__(self, transactions, min_support):
        self.root = FPNode(None, 1, None)
        self.min_support = min_support
        self.header_table = self.build_header_table(transactions)
        if self.header_table:
            for transaction in transactions:
                sorted_items = self.sort_items(transaction)
                if sorted_items:
                    self.insert_tree(sorted_items, self.root, 1)

    def build_header_table(self, transactions):
        item_counts = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1
        header_table = {item: [count, None] for item, count in item_counts.items() if count >= self.min_support}
        return header_table

    def sort_items(self, transaction):
        return sorted([item for item in transaction if item in self.header_table],
                      key=lambda item: self.header_table[item][0], reverse=True)

    def insert_tree(self, transaction, root, count):
        current_node = root
        for item in transaction:
            if item not in current_node.children:
                new_node = FPNode(item, count, current_node)
                current_node.children[item] = new_node
                if self.header_table[item][1] is None:
                    self.header_table[item][1] = new_node
                else:
                    node = self.header_table[item][1]
                    while node.link:
                        node = node.link
                    node.link = new_node
            else:
                current_node.children[item].increment(count)
            current_node = current_node.children[item]

    def tree_to_string(self, node):
        result = []
        for child in node.children.values():
            subtree = self.tree_to_string(child)
            result.append(f"<{child.item_name}:{child.count}" + (", " + subtree if subtree else "") + ">")
        return ", ".join(result)

    def build_conditional_tree(self, conditional_patterns):
        transactions = []
        for path, count in conditional_patterns:
            transactions.extend([path] * count)
        if not transactions:
            return None
        return FPTree(transactions, self.min_support)

    def mine_patterns(self):
        patterns = {}
        conditional_pattern_bases = {}
        conditional_fp_trees = {}
        frequent_patterns = {}

        items = sorted(self.header_table.items(), key=lambda x: x[1][0])
        for item, (support, node) in items:
            conditional_patterns = []
            while node is not None:
                path = []
                parent = node.parent
                while parent and parent.item_name is not None:
                    path.append(parent.item_name)
                    parent = parent.parent
                if path:
                    conditional_patterns.append((path[::-1], node.count))
                node = node.link

            conditional_pattern_bases[item] = conditional_patterns
            conditional_tree = self.build_conditional_tree(conditional_patterns)
            conditional_fp_trees[item] = self.tree_to_string(conditional_tree.root) if conditional_tree else ""

            if conditional_tree:
                _, _, subtree_patterns = conditional_tree.mine_patterns()
                for pattern, count in subtree_patterns.items():
                    full_pattern = tuple(list(pattern) + [item])
                    patterns[full_pattern] = count
                    frequent_patterns[full_pattern] = count

            patterns[(item,)] = support
            frequent_patterns[(item,)] = support

        return conditional_pattern_bases, conditional_fp_trees, frequent_patterns

# ===================== Association Rules =====================
def generate_association_rules(frequent_itemsets, min_confidence_percent):
    min_confidence = min_confidence_percent / 100.0
    rules = []
    itemset_support = {frozenset(itemset): support for itemset, support in frequent_itemsets}
    for itemset, support in frequent_itemsets:
        if len(itemset) > 1:
            items = list(itemset)
            for i in range(1, len(items)):
                for antecedent in combinations(items, i):
                    antecedent = frozenset(antecedent)
                    consequent = frozenset(itemset) - antecedent
                    if consequent and antecedent in itemset_support:
                        confidence = itemset_support[frozenset(itemset)] / itemset_support[antecedent]
                        if confidence >= min_confidence:
                            rules.append((set(antecedent), set(consequent), round(confidence*100, 2), support))
    return rules

# ===================== Streamlit App =====================
st.set_page_config(page_title="FP-Growth Frequent Itemsets", layout="wide")
st.title("FP-Growth Frequent Itemsets Mining")

st.markdown("""
Upload a Transaction CSV and discover Frequent Itemsets and Association Rules  
using the *FP-Growth* algorithm.
""")

uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data")
    st.dataframe(df.head())

    column_options = df.columns.tolist()
    item_col = st.selectbox("Select the column containing items per transaction:", column_options)

    transactions = df[item_col].dropna().astype(str).apply(lambda x: x.split(',')).tolist()

    min_support_percent = st.number_input("Enter Minimum Support (%)", min_value=1, max_value=100, value=5)
    min_confidence_percent = st.number_input("Enter Minimum Confidence (%)", min_value=1, max_value=100, value=60)

    total_transactions = len(transactions)
    min_support_count = math.ceil((min_support_percent / 100) * total_transactions)

    if st.button("Run FP-Growth"):
        fp_tree = FPTree(transactions, min_support_count)

        if not fp_tree.header_table:
            st.warning("No frequent items found with the given support.")
        else:
            _, _, frequent_patterns = fp_tree.mine_patterns()

            frequent_itemsets = [(set(pattern), support) for pattern, support in frequent_patterns.items()]

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
