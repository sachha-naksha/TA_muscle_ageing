import numpy as np, pandas as pd, scanpy as sc, matplotlib.pyplot as plt, os
from scipy.stats import hypergeom
import celloracle as co, glob, pickle
from functools import reduce
from tqdm import tqdm
import itertools, math, random
import networkx as nx

# ------------------------------------------------------------
# Functions for building networks and TF lists
# ------------------------------------------------------------
def create_combined_links_for_cluster_fusion(out_path, cluster_fusion, links_after_fit, quantile = 0.9):
    # Create combined links and finding threshold for edge strength
    combined_links = pd.DataFrame()
    for cluster in cluster_fusion:
        combined_links_og = links_after_fit[cluster]  # Convert to DataFrame
        combined_links_og['cluster'] = cluster
        combined_links = pd.concat([combined_links, combined_links_og], axis=0)
        combined_links[combined_links['cluster']==cluster]['coef_abs'].plot(kind='hist', bins=20, alpha=0.5)
    threshold = abs(combined_links['coef_abs']).quantile(quantile)
    print("Threshold for edge strength:", threshold)
    combined_links['strength'] = combined_links['coef_abs'].apply(lambda x: 0 if x<threshold else 1)
    print("strong edges count:", combined_links['strength'].value_counts())
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=1)
    plt.title(f'Threshold for edge strength: {threshold}')
    plt.xlabel('Absolute Coefficient Value')
    plt.ylabel('Frequency')
    plt.savefig(f"{out_path}/figures/combined_links_cutoff_histogram_{cluster_fusion}.pdf")
    plt.close()
    return combined_links, threshold

def filter_combined_links_and_build_grn(combined_links,threshold):
    # Filter combined links and create GRN
    # combined_links['strength'] = combined_links['coef_abs'].apply(lambda x: 0 if x<threshold else 1)
    grn = nx.MultiDiGraph()
    for _, row in combined_links.iterrows():
        grn.add_edge(
            row['source'],
            row['target'],
            weight=row['coef_mean'],
            cluster=row['cluster'],
            strength=row['strength'],
        )
    edges_df = pd.DataFrame([
        {
            'source': u,
            'target': v,
            'key': k,
            **d
        }
        for u, v, k, d in grn.edges(keys=True, data=True)
        ])
    return grn, edges_df

def filter_network_score_data(cluster_fusion, network_scores):
    combined_network_scores = pd.concat([network_scores[network_scores['cluster'] == int(cluster_id)] for cluster_id in cluster_fusion]) \
                                        .sort_values(by='degree_centrality_out', ascending=False) \
                                        .loc[lambda df: (~df.index.duplicated(keep='first'))]
    return combined_network_scores

# ------------------------------------------------------------
# Functions for doing enrichments
# ------------------------------------------------------------
def enrichment(P,p,S,s):
    # P is the total number of items in the population ===> Total starting genes in SLIDE
    # p is the number of successes in the population ===> Downstream genes
    # S is the sample size ===> Total SLIDE genes in LF
    # s is the number of successes in the sample ===> Common targets in LF
    p_value = 1 - hypergeom.cdf(s-1, P, p, S)   # Compute the p-value for observing X or more successes
    score = math.log2((s/S)/(p/P))
    return (score, p_value)

def get_SLIDE_GRN_enrichment(edges_df,cc_dict,cluster_fusion,ord_tf,slide_features,slide_starting_genes,total_source,case):
    possible_TF_combinations = list(itertools.combinations(total_source, ord_tf))
    strnth_cnd_TF_comb = list(itertools.product([0, 1], repeat=ord_tf))
    cc_dict.setdefault(cluster_fusion, {}).setdefault(ord_tf, [])
    for TF_comb in tqdm(possible_TF_combinations):
        # print(f"TF_comb: {TF_comb}")
        # FILTER1: Finding common targets from GRN for the TF_comb
        edges_grouped = edges_df[edges_df['source'].isin(TF_comb)].groupby(['source']).agg(list)
        if edges_grouped.empty:
            print(f"No edges found for TF_comb: {TF_comb}")
            continue
        common_targets_from_grn = set.intersection(*map(set, edges_grouped.target.values))
        for condition in strnth_cnd_TF_comb:
            if len(common_targets_from_grn) == 0:
                cc_dict[cluster_fusion][ord_tf].append([TF_comb, (condition), (0, 1), ([],[]), case, (pd.DataFrame(),pd.DataFrame())])
            else:
                # FILTER2: Vectorized filtering on (TF, condition) and common targets
                filter_df = pd.DataFrame({'source': TF_comb, 'strength': condition})
                filtered = edges_df.merge(filter_df, on=['source', 'strength'])
                filtered = filtered[filtered['target'].isin(common_targets_from_grn)]

                # Keep only targets regulated by all (TF, condition) pairs
                common_targets = (filtered.groupby('target')[['source', 'strength']].nunique().eq(len(filter_df)).all(axis=1))
                filtered_edges_df = filtered[filtered['target'].isin(common_targets[common_targets].index)]
                if filtered_edges_df.empty:
                    cc_dict[cluster_fusion][ord_tf].append([TF_comb, (condition), (0, 1), ([], []), case, (filtered_edges_df, pd.DataFrame())])
                else:
                    # Drop duplicates based on source and target, keeping the one with the maximum absolute weight
                    filtered_edges_df_unique = filtered_edges_df.loc[filtered_edges_df.groupby(['source', 'target'])['weight'].apply(lambda x: x.abs().idxmax())]
                    dwn_list = list(filtered_edges_df_unique['target'].unique())
                    dwngene = len(dwn_list)
                    # FILTER3: Finding intersection with latent features
                    cmn_list = list(set(filtered_edges_df_unique['target']).intersection(slide_features))
                    common = len(cmn_list)
                    if common == 0:
                        cc_dict[cluster_fusion][ord_tf].append([TF_comb, (condition), (0, 1), ([], [dwn_list]), case, (filtered_edges_df, filtered_edges_df_unique)])
                    else:
                        enrich = enrichment(slide_starting_genes, dwngene, len(slide_features), common)
                        cc_dict[cluster_fusion][ord_tf].append([TF_comb, (condition), enrich, (cmn_list, dwn_list), case, (filtered_edges_df,filtered_edges_df_unique)])
    return cc_dict

# ------------------------------------------------------------
# Functions for optionally generating the dataframe to be saved
# ------------------------------------------------------------
def _create_enrich_df(cc_dict, cluster_fusion, order_of_combination, filter):
    enrichment_df = pd.DataFrame(cc_dict[cluster_fusion][order_of_combination], columns=['TF', 'condition', 'ES', 'Genes', 'case', 'dfs'])
    enrichment_df[['score', 'p_value']] = pd.DataFrame(enrichment_df['ES'].tolist(), index=enrichment_df.index)
    enrichment_df[['common' , 'dwnstrm']] = pd.DataFrame(enrichment_df['Genes'].tolist(), index=enrichment_df.index)
    enrichment_df = enrichment_df.drop(columns=['ES', 'Genes'])
    filter_enrichment_df = (enrichment_df['condition'].isin(filter))
    enrichment_df = enrichment_df[filter_enrichment_df].sort_values(by='score', ascending=False)
    if order_of_combination == 1:
        non_zero_score_df = (enrichment_df['p_value']<0.05) \
                                & (enrichment_df['dwnstrm'].apply(len)>2) \
                                & (enrichment_df['common'].apply(len)>1) \
                                & (enrichment_df['score']>0) \
                                & (enrichment_df['case'].isin(['slide', 'net', 'rnd']))
        enrichment_df = enrichment_df[non_zero_score_df].reset_index(drop=True)
    elif order_of_combination == 2:
        non_zero_score_df = (enrichment_df['p_value']<0.05) \
                                & (enrichment_df['dwnstrm'].apply(len)>2) \
                                & (enrichment_df['common'].apply(len)>1) \
                                & (enrichment_df['score']>0) \
                                & (enrichment_df['case'].isin(['slide', 'net', 'rnd']))
        # Keep the zero scores for the cliffs delta analysis
        enrichment_df.loc[~non_zero_score_df, 'score'] = 0
        enrichment_df.loc[~non_zero_score_df, 'p_value'] = 1.0

    return enrichment_df

def create_enrichment_df(out_path, cc_dict, cluster_fusion, order_of_combination, filter = None, suffix=None, write = True):
    if order_of_combination ==1 and filter is None:
        filter = [(1,)]
    elif order_of_combination ==2 and filter is None:
        filter = [(1,1), (0,1), (1,0)]
    else:
        raise ValueError("Currently only supports order_of_combination= 1 or 2")
    enrichment_df = _create_enrich_df(cc_dict, cluster_fusion, order_of_combination, filter)
    if write:
        enrichment_df['dfs'] = enrichment_df['dfs'].apply(lambda x: x.to_json() if isinstance(x, pd.DataFrame) else str(x))
        enrichment_df.to_csv(f"{out_path}/out_files/SLIDE_LF_enrichment/enriched_df_{order_of_combination}_TFs_{cluster_fusion}{suffix}.csv", index=False)
    return enrichment_df

# ------------------------------------------------------------
# Functions for reading in the data
# ------------------------------------------------------------
def fetch_GRN_data(oracle_object_name, GRN_wd):
    # Read oracle links after fitting
    oracle = co.load_hdf5(f"{GRN_wd}/out_data/grn_inference/out_files/{oracle_object_name}")
    GRN_network_scores = pd.read_csv(f"{GRN_wd}/out_data/grn_inference/out_files/ridge_fitted_2_merged_network_scores.csv", index_col=0)
    GRN_TFs = oracle.all_regulatory_genes_in_TFdict
    GRN_links_after_fit = {key: [] for key in oracle.coef_matrix_per_cluster.keys()}
    for cluster in oracle.coef_matrix_per_cluster.keys():
        cluster_specific_links = oracle.coef_matrix_per_cluster[cluster].stack().reset_index()
        cluster_specific_links.columns = ['source', 'target', 'coef_mean']
        cluster_specific_links = cluster_specific_links[cluster_specific_links ['coef_mean'] != 0].reset_index(drop=True)
        cluster_specific_links['coef_abs'] = np.abs(cluster_specific_links['coef_mean'])
        GRN_links_after_fit[cluster] = cluster_specific_links
        # links_after_fit[cluster]['weight'] = links_after_fit[cluster]['coef_abs'] * links_after_fit[cluster]['-logp']
        # GRN_TFs = GRN_TFs + list(links_after_fit[cluster].source.unique())
    return GRN_links_after_fit, GRN_network_scores, GRN_TFs

