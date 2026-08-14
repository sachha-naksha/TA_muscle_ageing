import sys
import os
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import gseapy as gp
import decoupler as dc
import scanpy as sc

from pathlib import Path
from itertools import chain, repeat

# Plotting helpers live in pygenelab.plotting (single source of truth).
from pygenelab.plotting import visualize_deg_results, create_volcano_plot, plot_violin_box_combo, plot_pathway_dotplot

##### DEG ANALYSIS UTILS #####

def perform_deg_analysis(adata_subset, output_dir, layer='log1p_norm_cb', condition_column='condition', reference='WT', comparison='KO', 
                        sex_label=''):
    """
    Perform differential gene expression analysis between conditions
    
    Parameters:
    -----------
    adata_subset : AnnData object
        Subsetted data (e.g., adata_male or adata_female)
    output_dir : str
        Directory to save the results
    layer : str
        Counts layer to use for the analysis
    condition_column : str
        Column name in adata.obs that contains condition information
    reference : str
        Reference condition (e.g., 'WT')
    comparison : str
        Comparison condition (e.g., 'KO')
    sex_label : str
        Label for the sex subset ('Male' or 'Female')
    """
    
    print(f"=== Performing DEG Analysis: {sex_label} {comparison} vs {reference} ===")
    
    # Make a copy to avoid modifying original data
    adata_work = adata_subset.copy()
    
    # Ensure we have the right conditions
    conditions = adata_work.obs[condition_column].unique()
    print(f"Available conditions: {conditions}")
    
    if reference not in conditions or comparison not in conditions:
        print(f"Warning: {reference} or {comparison} not found in {condition_column}")
        return None, None
    
    # Filter to only include the two conditions of interest
    mask = adata_work.obs[condition_column].isin([reference, comparison])
    adata_filtered = adata_work[mask].copy()
    
    print(f"Cells in analysis - {reference}: {(adata_filtered.obs[condition_column] == reference).sum()}")
    print(f"Cells in analysis - {comparison}: {(adata_filtered.obs[condition_column] == comparison).sum()}")
    
    # Perform differential expression using Wilcoxon rank-sum test
    sc.tl.rank_genes_groups(
        adata_filtered,
        groupby=condition_column,
        groups=[comparison],  # Compare KO
        reference=reference,   # Against WT
        method='wilcoxon',
        pts=True,  # Calculate percentage of cells expressing the gene
        tie_correct=True,
        use_raw=False,
        layer=layer
    )
    
    # Extract results
    result_df = sc.get.rank_genes_groups_df(adata_filtered, group=comparison)
    
    # Add additional statistics
    result_df['log2FC'] = result_df['logfoldchanges']
    result_df['-log10(pval)'] = -np.log10(result_df['pvals'])
    result_df['significant'] = (result_df['pvals_adj'] < 0.05) & (np.abs(result_df['logfoldchanges']) > 0.5)
    
    print(f"Total genes analyzed: {len(result_df)}")
    print(f"Significantly upregulated genes (padj < 0.05, |log2FC| > 0.5): {(result_df['significant'] & (result_df['log2FC'] > 0)).sum()}")
    print(f"Significantly downregulated genes (padj < 0.05, |log2FC| > 0.5): {(result_df['significant'] & (result_df['log2FC'] < 0)).sum()}")
    
    return adata_filtered, result_df

# filter the degs using thresholds
def filter_degs(degs, log2fc_thresh=0.05, pval_thresh=0.01):
    filtered_degs = degs[(degs['logfoldchanges'].abs() > log2fc_thresh) & (degs['pvals_adj'] < pval_thresh)]
    # sort the rows by absolute logfoldchanges
    filtered_degs = filtered_degs.sort_values(by='logfoldchanges', key=lambda x: x.abs(), ascending=False)
    return filtered_degs

def get_common_degs(degs_males, degs_females):
    # get the names of the common DEGs between sexes (just names column)
    common_deg_names = degs_males[degs_males['names'].isin(degs_females['names'])]['names'].unique()
    # slice the rows of the common degs df from the males df
    common_deg_df_male = degs_males[degs_males['names'].isin(common_deg_names)]
    common_deg_df_female = degs_females[degs_females['names'].isin(common_deg_names)]
    common_degs = pd.merge(common_deg_df_male, common_deg_df_female, on='names', how='inner', suffixes=('_male', '_female'))
    return common_degs

def get_opposite_sign_degs(common_degs):
    # get the degs that have opposite signs in common male and female degs
    opposite_sign_degs = common_degs[common_degs['logfoldchanges_male'] * common_degs['logfoldchanges_female'] < 0]
    return opposite_sign_degs

def get_same_sign_degs(common_degs):
    # get the degs that have same signs in common male and female degs
    same_sign_degs = common_degs[common_degs['logfoldchanges_male'] * common_degs['logfoldchanges_female'] > 0]
    return same_sign_degs

def save_deg_results(result_df, sex_label, output_dir):
    """
    Save DEG results to CSV files
    """    
    # Filter for significant results
    sig_results = result_df[result_df['significant']].copy()
    #sort the rows by pvals_adj
    sig_results = sig_results.sort_values('pvals_adj')
    # separate the rows by upregualtion or downregulation and save to different files
    sig_results_up = sig_results[sig_results['log2FC'] > 0]
    sig_results_down = sig_results[sig_results['log2FC'] < 0]
    if len(sig_results_up) > 0:
        sig_results_file_up = f'{output_dir}/DEG_results_{sex_label.lower()}_up.csv'
        sig_results_up.to_csv(sig_results_file_up, index=False)
        print(f"Significant DEG results saved at: {sig_results_file_up}")
    if len(sig_results_down) > 0:
        sig_results_file_down = f'{output_dir}/DEG_results_{sex_label.lower()}_down.csv'
        sig_results_down.to_csv(sig_results_file_down, index=False)
        print(f"Significant DEG results saved at: {sig_results_file_down}")
    
    return sig_results

##### GSEA ANALYSIS UTILS #####

def convert_genes(gene_list, direction='human_to_mouse'):
    """
    Convert gene symbols between human and mouse formats.
    """
    direction = direction.lower()
    
    if direction in ['human_to_mouse', 'h2m']:
        # Human to mouse: capitalize (first letter uppercase, rest lowercase)
        return [gene.capitalize() for gene in gene_list]
    
    elif direction in ['mouse_to_human', 'm2h']:
        # Mouse to human: all uppercase
        return [gene.upper() for gene in gene_list]
    
    else:
        raise ValueError(
            f"Invalid direction: {direction}. "
            "Use 'human_to_mouse'/'h2m' or 'mouse_to_human'/'m2h'"
        )
        
def create_ranked_genelist(deg_df, log2fc_col='avg_log2FC', pval_col='p_val_adj', gene_col='gene_name', min_pval=1e-300):
    """
    Create a ranked gene list based on signed log2FC * -log10(adjusted p-value)
    """
    df = deg_df.copy()
    # Clip p-values
    df[pval_col] = df[pval_col].clip(lower=min_pval)
    # Calculate components and final score
    df['neg_log10_pval'] = -np.log10(df[pval_col])
    df['ranking_score'] = df[log2fc_col] * df['neg_log10_pval']
    # Sort by absolute ranking score
    ranked_df = df.sort_values('ranking_score', ascending=False)
    # Create final DataFrame with gene names as columns
    ranked_list = ranked_df.set_index(gene_col)[['ranking_score']].T
    return ranked_list

def simplify_term(term):
    # Convert to lowercase and split into words
    words = term.lower().replace('_', ' ').split()
    # Remove common words that don't add meaning
    stop_words = {'regulation', 'of', 'positive', 'negative', 'mediated', 'dependent', 
                 'activity', 'process', 'gobp', 'gomf', 'wp', 'reactome'}
    words = [w for w in words if w not in stop_words]
    return ' '.join(sorted(words))  # Sort words to match similar terms

def get_geneset_genes(df, msigdb_mice, output_file='geneset_genes.csv'):
    """
    Extract genes from genesets and save to CSV
    """
    # Create empty dictionary to store results
    geneset_genes = {}
    # For each pathway in the input dataframe
    for idx, row in df.iterrows():
        term = row['Term']
        # Get genes for this term from GSEA genesets (MSigDB)
        try:
            # Get leading edge genes if available
            genes = msigdb_mice[msigdb_mice['geneset'] == term]['genesymbol'].values
            geneset_genes[term] = {
                'Group': row['Group'],
                'Significance': row['-log10(pval)'],
                'Genes': ';'.join(genes)  # Join genes with semicolon for CSV
            }
        except KeyError:
            print(f"Warning: Could not find genes for {term}")
            continue
    
    # Convert to DataFrame
    result_df = pd.DataFrame.from_dict(geneset_genes, orient='index')
    result_df.index.name = 'Pathway'
    result_df.reset_index(inplace=True)
    
    # Save to CSV
    result_df.to_csv(output_file, index=False)
    
    return result_df

from pathlib import Path

def gmt_to_decoupler(pth: Path) -> pd.DataFrame:
    """Parse a gmt file to a decoupler pathway dataframe."""
    from itertools import chain, repeat

    pathways = {}

    with Path(pth).open("r") as f:
        for line in f:
            name, _, *genes = line.strip().split("\t")
            pathways[name] = genes

    return pd.DataFrame.from_records(
        chain.from_iterable(zip(repeat(k), v) for k, v in pathways.items()),
        columns=["geneset", "genesymbol"],
    )

def gmt_to_decoupler_multiple_pathways(gmt_paths, geneset_name=None, genesymbol_name=None):
    """Parse multiple gmt files and return a combined decoupler pathway dataframe."""
    all_records = []
    for pth in gmt_paths:
        with Path(pth).open("r") as f:
            for line in f:
                name, _, *genes = line.strip().split("\t")
                all_records.extend(zip(repeat(name), genes))
    return pd.DataFrame.from_records(all_records, columns=[geneset_name, genesymbol_name])
    
##### CELL TYPE ANALYSIS UTILS #####

# Calculate percentages for each condition
def get_cell_type_percentages(adata, cell_type_label='cell_type'):
    wt_cells = adata[adata.obs['condition'] == 'WT'].obs[cell_type_label].value_counts(normalize=True) * 100
    ko_cells = adata[adata.obs['condition'] == 'KO'].obs[cell_type_label].value_counts(normalize=True) * 100
    df = pd.DataFrame({
        'Cell Type': wt_cells.index,
        'WT%': wt_cells.values.round(2),
        'ΔERCC1 KO%': ko_cells.values.round(2)
    })
    return df

def get_cell_type_percentages_by_sex(adata, cell_type_label='cell_type'):
    # Calculate percentages for each combination of condition and sex
    wt_female = adata[(adata.obs['condition'] == 'WT') & (adata.obs['sex'] == 'F')].obs[cell_type_label].value_counts(normalize=True) * 100
    wt_male = adata[(adata.obs['condition'] == 'WT') & (adata.obs['sex'] == 'M')].obs[cell_type_label].value_counts(normalize=True) * 100
    ko_female = adata[(adata.obs['condition'] == 'KO') & (adata.obs['sex'] == 'F')].obs[cell_type_label].value_counts(normalize=True) * 100
    ko_male = adata[(adata.obs['condition'] == 'KO') & (adata.obs['sex'] == 'M')].obs[cell_type_label].value_counts(normalize=True) * 100
    df = pd.DataFrame({
        'Cell Type': wt_female.index,
        'WT F%': wt_female.values.round(2),
        'ΔERCC1 KO F%': ko_female.values.round(2),
        'WT M%': wt_male.values.round(2),
        'ΔERCC1 KO M%': ko_male.values.round(2)
    })
    return df

##### SCORE ANALYSIS UTILS #####

def calculate_sc_score(data, up_genes=None, down_genes=None, condition_col='condition'):
    """
    Calculate geneset signature scores for each sample/single-cell based on gene-sets with directionality
    
    Parameters:
    -----------
    data : pd.DataFrame or AnnData
        Expression data. If DataFrame, genes should be rows and samples columns.
        If AnnData, will be converted automatically (cells x genes -> genes x cells)
    up_genes : list, optional
        List of upregulated genes in the signature
    down_genes : list, optional
        List of downregulated genes in the signature
    condition_col : str, optional
        Name of the condition column in adata.obs to include in results (default: 'condition')
    """
    import scipy.sparse as sp
    
    if up_genes is None and down_genes is None:
        raise ValueError("At least one of up_genes or down_genes must be provided")
    
    # Store condition info if AnnData
    condition_info = None
    
    # Check if input is AnnData and convert to DataFrame
    if hasattr(data, 'X'):  # Check if it's an AnnData object
        print("Converting AnnData to DataFrame...")
        
        # Extract condition information before conversion
        if condition_col in data.obs.columns:
            condition_info = data.obs[condition_col].copy()
            print(f"Extracted '{condition_col}' column from obs")
        
        # Convert sparse matrix to dense if needed
        if sp.issparse(data.X):
            expr_matrix = data.X.toarray()
        else:
            expr_matrix = data.X
        
        # Create DataFrame with genes as rows, cells as columns (transpose)
        data = pd.DataFrame(
            expr_matrix.T,  # Transpose: genes x cells
            index=data.var_names,  # Gene names
            columns=data.obs_names  # Cell/sample names
        )
        print(f"Converted to DataFrame with shape: {data.shape} (genes x cells)")
    
    # Set index to gene names if not already done
    if "NAME" in data.columns:
        data = data.set_index("NAME")
    if "Description" in data.columns:
        data = data.drop("Description", axis=1)
    
    # Check which genes are present in the expression data
    available_genes = set(data.index)
    
    # Process up-regulated genes
    if up_genes is not None:
        up_genes_set = set(up_genes)
        up_genes = list(up_genes_set.intersection(available_genes))
        print(f"Using {len(up_genes)} upregulated genes")
    
    # Process down-regulated genes
    if down_genes is not None:
        down_genes_set = set(down_genes)
        down_genes = list(down_genes_set.intersection(available_genes))
        print(f"Using {len(down_genes)} downregulated genes")
    
    # Check if we have enough genes to proceed
    if (up_genes is None or len(up_genes) == 0) and (
        down_genes is None or len(down_genes) == 0
    ):
        raise ValueError(
            "No genes from the gene sets were found in the expression data"
        )
    
    sample_names = data.columns
    expr_matrix = data.select_dtypes(include=[np.number])
    
    # Z-standardize the expression values across samples
    z_standardized = (
        expr_matrix - expr_matrix.mean(axis=1).values.reshape(-1, 1)
    ) / expr_matrix.std(axis=1).values.reshape(-1, 1)
    
    # Calculate scores for each sample
    scores = pd.Series(0, index=sample_names)
    
    # Calculate total gene set size for normalization
    total_genes = 0
    if up_genes:
        total_genes += len(up_genes)
    if down_genes:
        total_genes += len(down_genes)
    
    # Calculate combined score with size normalization
    if up_genes:
        up_score = z_standardized.loc[up_genes].sum()
        scores += up_score
    if down_genes:
        down_score = z_standardized.loc[down_genes].sum()
        scores -= down_score  # Subtract because these are down-regulated
    
    # Normalize by square root of gene set size
    scores = scores / np.sqrt(total_genes)
    
    # Final z-score normalization across samples
    scores = (scores - scores.mean()) / scores.std()
    
    # Convert to DataFrame with meaningful column name
    scores_df = pd.DataFrame(scores, columns=["geneset_score"])
    
    # Add condition column if available
    if condition_info is not None:
        scores_df[condition_col] = condition_info.values
    
    return scores_df
    
def calculate_pairwise_significance(data, groups, x_var, y_var):
    """
    Calculate pairwise significance between all groups
    Returns a dictionary of p-values and significance levels
    """
    from scipy import stats
    results = {}
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            group1 = data[data[x_var] == groups[i]][y_var]  # Changed from 'category' and 'senescence_score'
            group2 = data[data[x_var] == groups[j]][y_var]  # Changed from 'category' and 'senescence_score'
            
            # Perform Mann-Whitney U test
            statistic, pvalue = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            
            # Add significance stars
            if pvalue < 0.001:
                sig = '***'
            elif pvalue < 0.01:
                sig = '**'
            elif pvalue < 0.05:
                sig = '*'
            else:
                sig = 'ns'
                
            results[(i, j)] = {'p-value': pvalue, 'significance': sig}
    
    return results

