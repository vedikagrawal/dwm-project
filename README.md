# FP-Growth Frequent Itemsets Mining
This project provides an interactive Streamlit web application that enables users to perform frequent itemset mining using the FP-Growth algorithm. It allows for the analysis of transactional data to discover frequent patterns and generate association rules.​

# Features
FP-Growth Algorithm: Efficiently mines frequent itemsets from transactional data.
Conditional Pattern Bases and FP-Trees: Visual representation of conditional patterns and their corresponding FP-Trees.
Association Rule Generation: Derives association rules from frequent itemsets with user-defined confidence thresholds.
Data Visualization: Displays frequent items and their support counts using bar charts.
Interactive Interface: User-friendly Streamlit interface for data upload, parameter selection, and result visualization.


# Demo
# Installation

## Clone the repository:

git clone https://github.com/vedikagrawal/dwm-project.git
cd dwm-project

## Clone the repository:Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

## Clone the repository:Install the required packages:

pip install -r requirements.txt

### Usage

## Prepare your transactional data:
Ensure your CSV file has a column where each row represents a transaction, and items are comma-separated. For example:

Transaction
milk,bread,eggs
bread,butter
milk,bread,butter,eggs

## Run the Streamlit application:

streamlit run fp.py

## Interact with the application:

Upload your CSV file.
Select the column containing transaction data.
Set the minimum support count and confidence percentage.
Click on "Run FP-Growth" to process the data.
View the generated frequent itemsets, conditional pattern bases, FP-Trees, and association rules.
Download the results as CSV files.​

# Dependencies

Python 3.7 or higher
Streamlit
pandas
matplotlib​

Install all dependencies using the provided requirements.txt file.​
GitHub

# Contributing
Contributions are welcome! If you have suggestions for improvements or encounter any issues, please open an issue or submit a pull request.​

# License
This project is licensed under the MIT License.​

# Acknowledgements
This project was developed as part of the Data Warehousing and Mining (DWM) course to demonstrate the application of the FP-Growth algorithm in mining frequent itemsets and generating association rules.​

