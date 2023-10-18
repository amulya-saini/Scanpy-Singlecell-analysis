# %%
#!pip install scanpy
#!pip install leidenalg

# %%
#importing required packages
import scanpy as sc
import pandas as pd
import numpy as np

# %%
#reading the data as an anndata (data is obtained from 10x genomics dataset collection)
adata = sc.read_10x_h5("C://Users//saini//OneDrive//Desktop//Project//5k_mouse_lung_CNIK_3pv3_filtered_feature_bc_matrix.h5")

# %%
#Making the variable names unique
adata.var_names_make_unique(join = "-") 

# %% [markdown]
# #Data intuition

# %%
#looking at all the variable from adata
adata.var

# %%
#observations present in adata
adata.obs

# %%
#the sparse matrix of adata
adata.X
#the data contains 7788 cells and 32285 genes

# %%
#all the gene names from adata
adata.var_names

# %%
adata

# %% [markdown]
# #preprocessing

# %%
sc.pl.highest_expr_genes(adata, n_top=20, )

# %%
#filtering cells based on number of genes in a cell
#filter out cells that have less than 300 genes
sc.pp.filter_cells(adata, min_genes = 300)

# %%
adata
#filtered out 99 cells

# %%
#filtering genes based on number of cells
# filter out genes that is seen in less than 5 cells
sc.pp.filter_genes(adata, min_cells = 5)

# %%
adata
#filtered out 11,123 genes

# %%
#checking for mitochondrial genes by checking if the varable name starts with "mt"
#and adding a new column to the data as "mt"
adata.var['mt'] = adata.var_names.str.startswith('mt-') 
#usually the mitochondrial genes are named with mt... in mouse data

# %%
#checking if the column is added
adata.var

# %%
#calling the variables that are true for mitochondrial genes
adata.var[adata.var.mt == True] 

# %%
#calculating the qc metrics for mt genes
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top = None, log1p = False, inplace = True)

# %%
adata.obs
#shows the total mitochondrial counts per cell and the percentage 

# %%
#violin plots of genes by counts, total counts, percentage of mt per cell
sc.pl.violin(adata, ["n_genes_by_counts", "total_counts","pct_counts_mt"], jitter = 0.4, multi_panel = True)

# %%
# scatter plot

#total counts vs percentage of mt gene
sc.pl.scatter(adata,x = "total_counts", y = "pct_counts_mt")
#total counts vs number of genes per cell
sc.pl.scatter(adata,x = "total_counts", y = "n_genes_by_counts")

# %%
#setting upper limit(98%) and lower limit(2%) for the number of genes by counts 
upper_lim = np.quantile(adata.obs.n_genes_by_counts.values, .98)
lower_lim = np.quantile(adata.obs.n_genes_by_counts.values, .02)
print(f'{lower_lim} to {upper_lim}')

#you can also filter the cells by manually looking at the graphs and picking a threshold
#adata = adata[adata.obs.n_genes_by_counts < 7000]

# %%
#filtering the adata based on the upper limit and lower limit
adata = adata[(adata.obs.n_genes_by_counts < upper_lim) & (adata.obs.n_genes_by_counts > lower_lim)]

# %%
adata
#filtered out 309 genes

# %%
#filtering genes that has more than 5% mt genes
adata = adata[adata.obs.pct_counts_mt < 5]

# %%
adata
#filtered out 453 genes

# %%
adata.X[2,:].sum()
#this shows how many UMI counts are associated to one gene

# %%
#Normalizing cells to 10,000 counts
sc.pp.normalize_total(adata, target_sum = 1e4)

# %%
adata.X[2,:].sum()
#the count is normalized to 10,000

# %%
#normalising the UMI counts to log counts
sc.pp.log1p(adata) 

# %%
adata.X[4,:].sum()

# %% [markdown]
# Feature selection

# %%
#this creates a new variable of highly_variable and adds false/true value
sc.pp.highly_variable_genes(adata)

# %%
adata.var

# %%
#calling only highly variable genes (True)
adata.var[adata.var.highly_variable]

# %%
#saving the data before processing any more values and further filtering
adata.raw = adata 

# %%
adata.raw.X

# %%
#Feature selection
#selecting genes that highly variable
adata = adata[:, adata.var.highly_variable]

# %%
adata

# %%
#regress out effects of total_counts per cell
sc.pp.regress_out(adata, ["total_counts" , "pct_counts_mt"]) 

# %%
#scaling each gene to unit variance
sc.pp.scale(adata, max_value = 10)

# %% [markdown]
# Principal component Analysis

# %%
#calculating the pca values
sc.tl.pca(adata, svd_solver = "arpack")

# %%
#creating a scatter plot in the PCA coordinated
sc.pl.pca(adata, color='Malat1')

# %%
#plotting the pca values
sc.pl.pca_variance_ratio(adata, log = True)

#the ones on the top are the reason for variance

# %%
#computing the neighborhood graph
sc.pp.neighbors(adata, n_neighbors = 10, n_pcs = 22)

# %%
#calculationg the umap
sc.tl.umap(adata) 

# %%
#plotting the umap
sc.pl.umap(adata) 

# %%
#!pip install leidenalg
#!pip install louvain
#!conda install -y -c anaconda cmake

# %%
#plotting the umap for selected genes
sc.pl.umap(adata, color = ['Malat1', 'Scgb1a1'])

# %% [markdown]
# Clustering

# %%
#running leiden clustering
sc.tl.leiden(adata, resolution = 0.09)

# %%
#plotiing the leiden cluster
sc.pl.umap(adata, color = ['leiden'])

# %% [markdown]
# ### find marker genes

# %%
#computing the top 15genes in each cluster
#comparing cells from one cluster to the rest of the cells
sc.tl.rank_genes_groups(adata, "leiden", method = "wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes = 15, sharey = False) 

# %%
#creating a dataframe of top 15genes in each cluster
top_genes = pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(15)

# %%
top_genes

# %%
top_genes.to_excel("C://Users//saini//OneDrive//Desktop//Project//top_15_genes.xlsx")

# %%
##plotting the gene expression along the clusters
sc.pl.umap(adata, color = ['leiden', 'Dock10'])

# %%
#if you want to compare certain gene expression across different clusters
sc.pl.violin(adata, ["Dock10"], groupby = "leiden")

# %%
#plotting the gene expression along the clusters
sc.pl.umap(adata, color = ['leiden', 'Ank3'])

# %%
sc.pl.violin(adata, ["Ank3"], groupby = "leiden")

# %% [markdown]
# To make a dataframe for highly expressed genes in a cluster

# %%
#dot plot showing the log fold changes
sc.pl.rank_genes_groups_dotplot(adata, n_genes=3, values_to_plot='logfoldchanges', min_logfoldchange=3, vmax=7, vmin=-7, cmap='bwr', groups = ['1'])

# %%
#dot plot showing the top 3 genes from each cluster
sc.pl.rank_genes_groups_dotplot(adata, n_genes=3)

# %%
adata.uns["rank_genes_groups"]['names']["0"]

#to get all the genes from the cluster you want use the above code
#you can get information about names, scores, pvalues, adjusted pvalues and log fold changes

# %%
#feeding the adata.uns output to results
results = adata.uns["rank_genes_groups"]
results["names"]["0"]

# %%
results["names"].dtype.names

# %%
#this will pullout the names, scores, pvals,adjusted pvals and log fold changes into an array
out = np.array([[0,0,0,0,0,0]])
for group in results["names"].dtype.names:
    out = np.vstack((out, np.vstack((results["names"][group],
                                    results["scores"][group],
                                     results["pvals"][group],
                                    results["pvals_adj"][group],
                                    results["logfoldchanges"][group],
                                    np.array([group] * len(results["names"][group])).astype("object"))).T))
out

# %%
#making a dataframe of all the gene markers
Gene_markers = pd.DataFrame(out[1:], columns = ["Gene" , "Scores" , "P_vales","Adjusted_P_values", "log_fold_changes" , "Cluster_Number"])
Gene_markers

# %%
markers = Gene_markers[(Gene_markers.Adjusted_P_values < 0.05) & (abs(Gene_markers.log_fold_changes) > 1)]
markers

# %%
#printing the table for the cells in cluster 6
markers[markers.Cluster_Number == "6"]

# %%
#finding the index of Dock10 gene (and the last two zeros make sure we feed only the number into the variable)
Dock10_index = np.where(adata.raw.var_names == "Dock10")[0][0]
Dock10_index

# %%
#this creates an array with value of Dock10 that is present in the every cell
D_cell = adata.raw.X.toarray()[:,Dock10_index]
D_cell

# %%
#adding a column to the adata observations and feeding the array values into it
#make sure to name the column differently from the gene name to avoid recepition of same value in the rows and columns
adata.obs["dock10"] = D_cell
adata.obs

# %%
#calling the cells that have dock10 more than 0
adata.obs[adata.obs.dock10 > 0]

# %%



