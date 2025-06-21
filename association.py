from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

def run_association_mining(df):
    # Convert all values to strings for transactional format
    transactions = df.astype(str).values.tolist()

    # Use TransactionEncoder to binarize
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)

    # Apply Apriori with lower support
    freq_items = apriori(df_trans, min_support=0.05, use_colnames=True)

    if freq_items.empty:
        return "<p><b>No frequent itemsets found.</b> Try lowering the support threshold or using categorical data.</p>"

    rules = association_rules(freq_items, metric="lift", min_threshold=1.0)

    if rules.empty:
        return "<p><b>No strong association rules found.</b></p>"

    return rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head().to_html()
