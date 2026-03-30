import ast
import glob
import itertools
import math
import os
import pickle
import random
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scanpy as sc
from scipy.stats import hypergeom
from tqdm import tqdm

import celloracle as co

from utils import (
    create_combined_links_for_cluster_fusion,
    filter_combined_links_and_build_grn,
    filter_network_score_data,
    enrichment,
    get_SLIDE_GRN_enrichment,
    create_enrichment_df,
    _create_enrich_df,
)


class StateSpecificEnrichment:
    """
      1. load GRN + SLIDE data
      2. build combined links per cluster fusion
      3. construct TF lists (SLIDE-informed, network-matched, random)
      4. run hypergeometric enrichment
      5. post-process and merge weights
      6. visualise results (bar charts, TF-centric networks)
    """

    def __init__(
        self,
        grn_wd: str,
        oracle_object_name: str,
        feature_folder: str,
        out_path: str,
        experiment: str,
        slide_starting_genes: int,
        clusters_of_interest: list[str],
        order_fr_clust: list[int],
        order_fr_tfcomb: list[int],
        weight: str = "strength",
        quantile: float = 0.70,
        seed: int = 42,
    ):
        self.grn_wd = grn_wd
        self.oracle_object_name = oracle_object_name
        self.feature_folder = feature_folder
        self.out_path = out_path
        self.experiment = experiment
        self.slide_starting_genes = slide_starting_genes
        self.clusters_of_interest = clusters_of_interest
        self.order_fr_clust = order_fr_clust
        self.order_fr_tfcomb = order_fr_tfcomb
        self.weight = weight
        self.quantile = quantile

        random.seed(seed)
        os.makedirs(f"{out_path}/figures", exist_ok=True)
        os.makedirs(f"{out_path}/out_files", exist_ok=True)
        os.makedirs(f"{out_path}/out_files/SLIDE_LF_enrichment", exist_ok=True)
        sc.settings.figdir = f"{out_path}/figures"
        plt.rcParams["figure.figsize"] = [6, 4.5]
        plt.rcParams["savefig.dpi"] = 300

        self.GRN_links_after_fit = None
        self.GRN_network_scores = None
        self.GRN_TFs = None
        self.slide_features = None
        self.cluster_fusions = []
        self.cc_dicts: dict[tuple, dict[int, dict]] = {}
        self.enrichment_dfs: dict[tuple, dict[int, pd.DataFrame]] = {}

    # ── 1. Data loading ────────────────────────────────────────────────────

    def load_grn_data(self):
        oracle = co.load_hdf5(
            f"{self.grn_wd}/out_files/{self.oracle_object_name}"
        )
        self.GRN_network_scores = pd.read_csv(
            f"{self.grn_wd}/out_files/ridge_fitted_2_merged_network_scores.csv",
            index_col=0,
        )
        self.GRN_TFs = oracle.all_regulatory_genes_in_TFdict
        self.GRN_links_after_fit = {
            key: [] for key in oracle.coef_matrix_per_cluster.keys()
        }
        for cluster in oracle.coef_matrix_per_cluster.keys():
            links = (
                oracle.coef_matrix_per_cluster[cluster]
                .stack()
                .reset_index()
            )
            links.columns = ["source", "target", "coef_mean"]
            links = links[links["coef_mean"] != 0].reset_index(drop=True)
            links["coef_abs"] = np.abs(links["coef_mean"])
            self.GRN_links_after_fit[cluster] = links

    def load_slide_features(self):
        feature_files = glob.glob(f"{self.feature_folder}/*feature_list*")
        if self.experiment == "male_type2":
            data = [
                pd.read_csv(f, sep="\t", header=0)
                for f in feature_files
                if "Z15" in f or "Z29" in f
            ]
        elif self.experiment == "female_type2":
            data = [
                pd.read_csv(f, sep="\t", header=0)
                for f in feature_files
                if "Z15" in f or "Z12" in f
            ]
        else:
            raise ValueError(f"Unknown experiment: {self.experiment}")
        self.slide_features = set(pd.concat(data)["names"])

    def load_slide_features_with_corr(self):
        feature_files = glob.glob(f"{self.feature_folder}/*feature_list*")
        if self.experiment == "male_type2":
            data = [
                pd.read_csv(f, sep="\t", header=0)
                for f in feature_files
                if "Z15" in f or "Z29" in f
            ]
        elif self.experiment == "female_type2":
            data = [
                pd.read_csv(f, sep="\t", header=0)
                for f in feature_files
                if "Z15" in f or "Z38" in f or "Z42" in f or "Z55" in f
            ]
        else:
            raise ValueError(f"Unknown experiment: {self.experiment}")
        df = pd.concat(data)
        self.positive_corr_names = set(df[df["corrs"] >= 0]["names"])
        self.negative_corr_names = set(df[df["corrs"] < 0]["names"])

    def load_slide_features_from_files(self, feature_files: list[str] | None = None, a_loading_threshold: float | None = None):
        if feature_files is None:
            feature_files = glob.glob(f"{self.feature_folder}/*feature_list*")
        data = pd.concat([pd.read_csv(f, sep="\t", header=0) for f in feature_files])
        if a_loading_threshold is not None:
            data = data[data["A_loading"] >= a_loading_threshold]
        self.slide_features = set(data["names"])

    # ── 2. Run enrichment ──────────────────────────────────────────────────

    def build_cluster_fusions(self):
        self.cluster_fusions = []
        for ord_clus in self.order_fr_clust:
            self.cluster_fusions += list(
                itertools.combinations(self.clusters_of_interest, ord_clus)
            )

    def run_enrichment(self, save_pickle: bool = True):
        self.build_cluster_fusions()
        for cluster_fusion in self.cluster_fusions:
            combined_links, threshold = create_combined_links_for_cluster_fusion(
                self.out_path,
                cluster_fusion,
                self.GRN_links_after_fit,
                quantile=self.quantile,
            )
            grn, edges_df = filter_combined_links_and_build_grn(
                combined_links, threshold
            )

            fig = (
                edges_df.groupby(["strength", "key"])
                .size()
                .unstack()
                .plot(kind="bar", stacked=True)
            )
            plt.xlabel("Strength")
            plt.ylabel("Count")
            plt.title("Distribution of Key and Strength")
            plt.savefig(
                f"{self.out_path}/figures/combined_links_key_strength_{cluster_fusion}{self.experiment}.pdf"
            )
            plt.close()

            slide_features = self.slide_features.intersection(set(grn.nodes))
            slide_features_neighbors = []
            for gene in slide_features:
                slide_features_neighbors += list(grn.predecessors(gene))
            slide_tot_TF = (
                slide_features.union(set(slide_features_neighbors))
            ).intersection(self.GRN_TFs)

            for ord_tf in self.order_fr_tfcomb:
                cc_dict = {}
                if ord_tf == 1:
                    cases = [(slide_tot_TF, "slide")]
                elif ord_tf == 2:
                    cases = [(slide_tot_TF, "slide")]
                else:
                    raise ValueError(f"Unsupported TF combination order: {ord_tf}")

                for TFs, case in cases:
                    print(
                        f"Running enrichment for {self.experiment}, "
                        f"{cluster_fusion}, {ord_tf} TFs, case = {case}"
                    )
                    cc_dict = get_SLIDE_GRN_enrichment(
                        edges_df,
                        cc_dict,
                        cluster_fusion,
                        ord_tf,
                        slide_features,
                        self.slide_starting_genes,
                        TFs,
                        case,
                    )

                self.cc_dicts.setdefault(cluster_fusion, {})[ord_tf] = cc_dict

                if save_pickle:
                    suffix = f"_{self.experiment}"
                    print(
                        f"Creating enrichment dataframe for {self.experiment}, "
                        f"{cluster_fusion}, {ord_tf} TFs, all cases"
                    )
                    enrichment_df = create_enrichment_df(
                        self.out_path,
                        cc_dict,
                        cluster_fusion,
                        ord_tf,
                        filter=None,
                        suffix=suffix,
                    )
                    self.enrichment_dfs.setdefault(cluster_fusion, {})[
                        ord_tf
                    ] = enrichment_df
                    print(enrichment_df["case"].value_counts())
                    pkl_path = (
                        f"{self.out_path}/out_files/SLIDE_LF_enrichment/"
                        f"cc_dict_{ord_tf}_TFs_{cluster_fusion}{suffix}.pickle"
                    )
                    with open(pkl_path, "wb") as f:
                        pickle.dump(cc_dict, f)

    # ── 3. Post-processing ─────────────────────────────────────────────────

    def load_results(self, cluster_fusion: tuple, ord_tf: int):
        suffix = f"_{self.experiment}"
        pkl_path = (
            f"{self.out_path}/out_files/SLIDE_LF_enrichment/"
            f"cc_dict_{ord_tf}_TFs_{cluster_fusion}{suffix}.pickle"
        )
        with open(pkl_path, "rb") as f:
            cc_dict = pickle.load(f)
        self.cc_dicts.setdefault(cluster_fusion, {})[ord_tf] = cc_dict

        csv_path = (
            f"{self.out_path}/out_files/SLIDE_LF_enrichment/"
            f"enriched_df_{ord_tf}_TFs_{cluster_fusion}{suffix}.csv"
        )
        self.enrichment_dfs.setdefault(cluster_fusion, {})[ord_tf] = pd.read_csv(
            csv_path
        )

    def build_used_weight_df(self, cluster_fusion: tuple, ord_tf: int) -> pd.DataFrame:
        suffix = f"_{self.experiment}"
        cc_dict = self.cc_dicts[cluster_fusion][ord_tf]
        concatenated = pd.DataFrame()
        for i in range(len(cc_dict[cluster_fusion][ord_tf])):
            w_df = cc_dict[cluster_fusion][ord_tf][i][5][1]
            if w_df.empty:
                continue
            w_df["case"] = cc_dict[cluster_fusion][ord_tf][i][4]
            concatenated = pd.concat([concatenated, w_df], ignore_index=True)
        out = (
            f"{self.out_path}/out_files/SLIDE_LF_enrichment/"
            f"{ord_tf}_TFs_{cluster_fusion}{suffix}_used_weights.csv"
        )
        concatenated.to_csv(out, index=False)
        return concatenated

    def build_slide_lf_enriched(
        self, cluster_fusion: tuple, ord_tf: int
    ) -> pd.DataFrame:
        suffix = f"_{self.experiment}"
        enr_path = (
            f"{self.out_path}/out_files/SLIDE_LF_enrichment/"
            f"enriched_df_{ord_tf}_TFs_{cluster_fusion}{suffix}.csv"
        )
        wt_path = (
            f"{self.out_path}/out_files/SLIDE_LF_enrichment/"
            f"{ord_tf}_TFs_{cluster_fusion}{suffix}_used_weights.csv"
        )
        slide_lf = pd.read_csv(enr_path)
        used_w = pd.read_csv(wt_path)
        slide_lf = slide_lf[slide_lf["case"] == "slide"]
        used_w = used_w[used_w["case"] == "slide"]

        slide_lf = slide_lf[["TF", "common"]]
        slide_lf["TF"] = slide_lf["TF"].apply(ast.literal_eval).apply(tuple)
        slide_lf["common"] = slide_lf["common"].apply(ast.literal_eval).apply(list)
        slide_lf = slide_lf.explode("common").explode("TF")
        slide_lf.columns = ["source", "target"]
        used_w = used_w[used_w["strength"] == 1]
        slide_lf = slide_lf.merge(used_w, on=["source", "target"], how="left")

        out = (
            f"{self.out_path}/out_files/SLIDE_LF_enrichment/"
            f"enriched_df_{ord_tf}_TFs_{cluster_fusion}{suffix}_used_weights_strong.csv"
        )
        slide_lf.to_csv(out, index=False)
        return slide_lf

    # ── 4. Visualization ───────────────────────────────────────────────────

    def plot_strength_key_distribution(
        self, slide_lf_enriched: pd.DataFrame, cluster_fusion: tuple, ord_tf: int
    ):
        suffix = f"_{self.experiment}"
        slide_lf_enriched.groupby(["strength", "key"]).size().unstack().plot(
            kind="bar", stacked=True
        )
        plt.xlabel("Strength")
        plt.ylabel("Count")
        plt.title("Distribution of Key and Strength")
        plt.savefig(
            f"{self.out_path}/figures/"
            f"enriched_key_strength_{ord_tf}_TFs_{cluster_fusion}{suffix}_used_weights_strong.pdf"
        )
        plt.close()

    def plot_enrichment_scores_with_color_proportions(
        self,
        slide_lf_enriched: pd.DataFrame,
        cluster_fusion: tuple,
        ord_tf: int,
    ):
        if not hasattr(self, "positive_corr_names"):
            self.load_slide_features_with_corr()

        suffix = f"_{self.experiment}"
        cc_dict = self.cc_dicts[cluster_fusion][ord_tf]

        lf = slide_lf_enriched.copy()
        lf["color"] = lf["target"].apply(
            lambda g: "red"
            if g in self.positive_corr_names
            else ("blue" if g in self.negative_corr_names else "gray")
        )
        color_hist = (
            lf.groupby("source")["color"]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
        )

        tf_name_score = {k: None for k in color_hist.index.values}
        for entry in cc_dict[cluster_fusion][ord_tf]:
            tf, cond, score_val = entry[0][0], entry[1], entry[2][0]
            if tf in tf_name_score and cond == (1,):
                tf_name_score[tf] = score_val

        sorted_data = sorted(
            [(v, k) for k, v in tf_name_score.items() if v is not None],
            reverse=True,
            key=lambda x: x[0],
        )
        if not sorted_data:
            raise ValueError("No valid TF scores found for plotting.")

        score, tf_name = zip(*sorted_data)
        score_series = pd.Series(score, index=tf_name, name="score")
        color_proportions = color_hist.loc[list(tf_name)]
        original_props = color_proportions.copy()
        scaled = color_proportions.mul(score_series, axis=0).reset_index()
        scaled = scaled.rename(columns={"index": "source"})

        melted_scaled = scaled.melt(id_vars="source", var_name="color", value_name="height")
        melted_props = (
            original_props.reset_index().melt(
                id_vars="source", var_name="color", value_name="proportion"
            )
        )
        plot_df = melted_scaled.merge(melted_props, on=["source", "color"])
        plot_df["text"] = (plot_df["proportion"] * 100).round(1).astype(str) + "%"

        fig = px.bar(
            plot_df,
            x="source",
            y="height",
            color="color",
            text="text",
            labels={"height": "Score-scaled Proportion", "source": "TF"},
            title=(
                f"TF Enrichment Scores with color Proportions for "
                f"{self.experiment} - {cluster_fusion} - {ord_tf} Order"
            ),
        )
        fig.update_traces(textposition="inside", insidetextanchor="middle")
        fig.update_layout(
            xaxis_title="Transcription Factor (TF)",
            yaxis_title="Enrichment Score",
            xaxis_tickangle=-90,
            font=dict(family="Arial", size=8, color="black"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="white",
            xaxis=dict(showgrid=False, showline=True, linecolor="black", ticks="outside"),
            yaxis=dict(showgrid=False, showline=True, linecolor="black", ticks="outside"),
        )
        fig.write_image(
            f"{self.out_path}/figures/"
            f"0_7_TF_Enrichment_Scores_with_color_Proportions_{ord_tf}_TFs_{cluster_fusion}_{self.experiment}.svg",
            format="svg",
        )
        return fig
