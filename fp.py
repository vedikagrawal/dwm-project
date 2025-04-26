import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
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
        self.min_support = min_support
        self.root, self.header_table = self.construct_fp_tree(transactions)

    def construct_fp_tree(self, transactions):
        header_table = self.build_header_table(transactions)
        if len(header_table) == 0:
            return None, None
        root = FPNode(None, 1, None)
        for transaction in transactions:
            sorted_items = self.sort_items(transaction, header_table)
            if sorted_items:
                self.insert_tree(sorted_items, root, header_table, 1)
        return root, header_table

    def build_header_table(self, transactions):
        item_counts = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1
        return {item: [cnt, None] for item, cnt in item_counts.items() if cnt >= self.min_support}

    def sort_items(self, transaction, header_table):
        return sorted([item for item in transaction if item in header_table],
                      key=lambda item: header_table[item][0], reverse=True)

    def insert_tree(self, transaction, root, header_table, count):
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

    def build_conditional_tree(self, conditional_patterns):
        item_counts = defaultdict(int)
        for path, count in conditional_patterns:
            for item in path:
                item_counts[item] += count

        # Now filter items by min_support
        item_counts = {item: cnt for item, cnt in item_counts.items() if cnt >= self.min_support}

        if not item_counts:
            return None

        transactions = []
        for path, count in conditional_patterns:
            filtered_path = [item for item in path if item in item_counts]
            filtered_path.sort(key=lambda item: item_counts[item], reverse=True)
            if filtered_path:
                transactions.append(filtered_path * count)

        # Flatten transactions
        flattened = []
        for t in transactions:
            flattened.append(t)

        return FPTree(flattened, self.min_support)

    def mine_patterns(self, suffix=None):
        if suffix is None:
            suffix = []

        patterns = {}
        items = sorted(self.header_table.items(), key=lambda x: x[1][0])  # sort by support ascending

        for item, (support, node) in items:
            new_suffix = suffix + [item]
            patterns[tuple(new_suffix)] = support

            conditional_patterns = []
            while node:
                path = []
                parent = node.parent
                while parent and parent.item_name:
                    path.append(parent.item_name)
                    parent = parent.parent
                if path:
                    conditional_patterns.append((path[::-1], node.count))
                node = node.link

            conditional_tree = self.build_conditional_tree(conditional_patterns)

            if conditional_tree and conditional_tree.root:
                subtree_patterns = conditional_tree.mine_patterns(new_suffix)
                patterns.update(subtree_patterns)

        return patterns

# ===================== Association Rules =====================
from itertools import combinations

def generate_association_rules(frequent_patterns, min_confidence_percent):
    min_confidence = min_confidence_percent / 100.0
    rules = []
    itemset_support = {frozenset(itemset): support for itemset, support in frequent_patterns.items()}

    for itemset in itemset_support:
        if len(itemset) > 1:
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    if consequent and antecedent in itemset_support:
                        confidence = itemset_support[itemset] / itemset_support[antecedent]
                        if confidence >= min_confidence:
                            rules.append((set(antecedent), set(consequent), round(confidence*100, 2), itemset_support[itemset]))

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
        tree = FPTree(transactions, min_support_count)

        if tree.root is None:
            st.warning("No frequent items found with the given support.")
        else:
            conditional_pattern_bases, conditional_fp_trees, frequent_patterns = tree.mine_patterns()

            if not frequent_patterns:
                st.warning("No frequent itemsets found.")
            else:
                st.success(f"Found {len(frequent_patterns)} frequent itemsets!")
                
                result_df = pd.DataFrame([(list(items), support) for items, support in frequent_patterns.items()],
                                         columns=["Itemset", "Support"])

                st.write("### Frequent Itemsets")
                st.dataframe(result_df)

                item_count = defaultdict(int)
                for items, support in frequent_patterns.items():
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

            rules = generate_association_rules(frequent_patterns, min_confidence_percent)

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
