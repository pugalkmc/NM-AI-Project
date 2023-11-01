import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df = pd.read_excel('preprocessed_data.xlsx')

basket_sets = df.groupby(['BillNo', 'Itemname'])['Quantity'].sum().unstack().fillna(0)
basket_sets = (basket_sets > 0).astype(int)

frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)

association_rules_df = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

print(association_rules_df)
