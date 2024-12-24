import pandas as pd

df = pd.read_csv("D:/bjfu/GNN-transformer/protein_cluster/data/Pfam-zhushi/output_labels_top-10.tsv", sep='\t')
print(df.head())
print(df.sum())
