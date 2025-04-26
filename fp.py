
import streamlit as st
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

# ===================== FP-Growth Code (From Your Notebook) =====================
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


def build_header_table(transactions, min_support):
    item_counts = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_counts[item] += 1
    return {item: [count, None] for item, count in item_counts.items() if count >= min_support}


def sort_items(transaction, header_table):
    return sorted([item for item in transaction if item in header_table],
                  key=lambda item: header_table[item][0], reverse=True)


def insert_tree(transaction, root, header_table):
    current_node = root
    for item in transaction:
        if item not in current_node.children:
            new_node = FPNode(item, 1, current_node)
            current_node.children[item] = new_node
            if header_table[item][1] is None:
                header_table[item][1] = new_node
            else:
                node = header_table[item][1]
                while node.link:
                    node = node.link
                node.link = new_node
        else:
            current_node.children[item].increment(1)
        current_node = current_node.children[item]


def construct_fp_tree(transactions, min_support):
    header_table = build_header_table(transactions, min_support)
    if len(header_table) == 0:
        return None, None
    root = FPNode(None, 1, None)
    for transaction in transactions:
        sorted_items = sort_items(transaction, header_table)
        insert_tree(sorted_items, root, header_table)
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


def mine_fp_tree(header_table, prefix, frequent_itemsets, min_support):
    sorted_items = sorted(header_table.items(), key=lambda x: x[1][0])
    for base_item, (count, node) in sorted_items:
        new_freq_set = prefix.copy()
        new_freq_set.add(base_item)
        frequent_itemsets.append((new_freq_set, count))
        conditional_patterns = find_prefix_paths(base_item, node)
        conditional_transactions = []
        for path, count in conditional_patterns:
            conditional_transactions.extend([path] * count)
        conditional_tree, new_header_table = construct_fp_tree(conditional_transactions, min_support)
        if new_header_table:
            mine_fp_tree(new_header_table, new_freq_set, frequent_itemsets, min_support)

# ===================== Streamlit App =====================
st.set_page_config(page_title="Market Basket Analysis - FP-Growth", layout="wide")
st.title("🛒 Market Basket Analysis using FP-Growth")

st.markdown("""
This app allows you to upload a transaction dataset and run the **FP-Growth** algorithm to find frequent itemsets.
""")

# Upload CSV file
uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data")
    st.dataframe(df.head())

    # Ask user to select the column that contains items
    column_options = df.columns.tolist()
    item_col = st.selectbox("Select the column containing the list of items per transaction:", column_options)

    # Convert the item column to list of lists
    transactions = df[item_col].dropna().astype(str).apply(lambda x: x.split(',')).tolist()

    min_sup = st.number_input("Minimum Support (count)", min_value=1, value=2)

    if st.button("Run FP-Growth"):
        root, header_table = construct_fp_tree(transactions, min_sup)

        if root is None:
            st.warning("No frequent items found with the given support.")
        else:
            frequent_itemsets = []
            mine_fp_tree(header_table, set(), frequent_itemsets, min_sup)

            # Convert to DataFrame
            result_df = pd.DataFrame([(list(items), support) for items, support in frequent_itemsets],
                                     columns=["Itemset", "Support"])

            st.success(f"Found {len(result_df)} frequent itemsets!")
            st.write("### Frequent Itemsets")
            st.dataframe(result_df)

            # Plot top 10 frequent items
            item_count = defaultdict(int)
            for items, support in frequent_itemsets:
                for item in items:
                    item_count[item] += support

            top_items = sorted(item_count.items(), key=lambda x: x[1], reverse=True)[:10]
            item_names = [x[0] for x in top_items]
            item_support = [x[1] for x in top_items]

            fig, ax = plt.subplots()
            ax.bar(item_names, item_support, color='skyblue')
            plt.xticks(rotation=45)
            plt.title("Top Frequent Items")
            st.pyplot(fig)

            st.download_button("Download Results", result_df.to_csv(index=False), file_name="frequent_itemsets.csv")
