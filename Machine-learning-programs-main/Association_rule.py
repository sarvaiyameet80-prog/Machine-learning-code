import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

data = [
    ['milk', 'bread', 'egg'],
    ['milk', 'bread'],
    ['milk', 'egg'],
    ['bread', 'butter'],
    ['milk', 'bread', 'butter'],
    ['bread', 'egg'],
    ['milk', 'butter'],
    ['egg', 'butter'],
    ['milk', 'bread', 'egg', 'butter']
]

te = TransactionEncoder()
te_array = te.fit_transform(data)

df = pd.DataFrame(te_array, columns=te.columns_)

print("Encoded Dataset:\n")
print(df.head())


frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)

print("\nFrequent Itemsets:\n")
print(frequent_itemsets.sort_values(by="support", ascending=False))

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)


rules = rules.sort_values(by="lift", ascending=False)

print("\nAssociation Rules:\n")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

print("\nTop Rules:\n")
for i in range(len(rules)):
    print(f"Rule {i+1}:")
    print(f"{list(rules.iloc[i]['antecedents'])} → {list(rules.iloc[i]['consequents'])}")
    print(f"Support: {rules.iloc[i]['support']:.2f}")
    print(f"Confidence: {rules.iloc[i]['confidence']:.2f}")
    print(f"Lift: {rules.iloc[i]['lift']:.2f}")
    print("-" * 40)

import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(data=rules, x="support", y="confidence", size="lift")
plt.title("Association Rules Visualization")
plt.show()
