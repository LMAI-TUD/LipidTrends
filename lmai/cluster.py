# -*- coding: utf-8 -*-
#
# Copyright (C) 2021-2024  LMAI_team @ TU Dresden:
#     LMAI_team: Zhixu Ni, Maria Fedorova
#
# Licensing:
# This code is licensed under AGPL-3.0 license (Affero General Public License v3.0).
# For more information, please read:
#     AGPL-3.0 License: https://www.gnu.org/licenses/agpl-3.0.en.html
#
# Citation:
# Please cite our publication in an appropriate form.
#
# For more information, please contact:
#     Fedorova Lab (#LMAI_team): https://fedorovalab.net/
#     LMAI on Github: https://github.com/LMAI-TUD
#
import json
import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from sklearn import mixture
from sklearn.cluster import AgglomerativeClustering, BisectingKMeans, KMeans


def create_normalized_data(
        data, mode="zscore"
):
    scaled_df = data.copy()

    # zscore
    ds_file_np = scaled_df.to_numpy()
    scaled_df = pd.DataFrame(
        data=ds_file_np, index=scaled_df.index, columns=list(scaled_df)
    )
    if mode == "zscore":
        scaled_df = scaled_df.apply(stats.zscore, 1, False, "broadcast")
    elif mode == "log2":
        scaled_df = scaled_df.apply(lambda x: np.log2(x + 1), 1, False, "broadcast")
    elif mode == "min_max":
        scaled_df = scaled_df.apply(
            lambda x: (x - x.mean()) / x.std(), 1, False, "broadcast"
        )

    return scaled_df


def run_cluster(
        data,
        k: int = 10,
        method: str = "gmm",
        random_state: int = 0,
):
    """
    runs clusters on data with given k
    :param random_state:
    :param incl_median_var:
    :param data: pandas data frame.
    :param k: init k, default is set to 10.
    :param method: default is set to "gmm". all methods: ["km", "bisect_km", "gmm", "dpgmm"].
    :return: data frame with assigned clusters
    """

    if len(data.index) < k:
        k = len(data.index)
    else:
        pass

    if method is None:
        method = "gmm"

    new_col_dct = {}

    temp_df = data.copy()

    # drop nan rows from DataFrame
    temp_df.dropna(axis=0, how="any", inplace=True)

    # hierarchical clustering with metric = euclidean + ward linkage
    if method == "hew":
        hew = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward")
        hew.fit_predict(temp_df)
        new_col_dct["hew_cluster"] = hew.labels_

    # kmeans
    elif method == "km":
        kmeans = KMeans(
            init="k-means++", n_clusters=k, n_init="auto", random_state=random_state
        )
        kmeans.fit_predict(temp_df)
        new_col_dct["km_cluster"] = kmeans.labels_

    # bisecting kmeans
    elif method == "bisect_km":
        bisect_km = BisectingKMeans(n_clusters=k, random_state=random_state)
        bisect_km.fit_predict(temp_df)
        new_col_dct["bisect_km_cluster"] = bisect_km.labels_

    # Gaussians
    elif method == "gmm":
        gmm = mixture.GaussianMixture(
            n_components=k, covariance_type="full", random_state=random_state
        )
        gmm.fit(temp_df)
        gmm_labels = gmm.predict(temp_df)
        new_col_dct["gmm_cluster"] = gmm_labels

    # Fit a Dirichlet process mixture of Gaussians / Bayesian Gaussian mixture models with a Dirichlet process
    elif method == "dpgmm":
        dpgmm = mixture.BayesianGaussianMixture(
            n_components=k, covariance_type="full", random_state=random_state
        )
        dpgmm.fit(temp_df)
        dpgmm_labels = dpgmm.predict(temp_df)
        new_col_dct["dpgmm_cluster"] = dpgmm_labels
    else:
        pass

    for method_key, method_val in new_col_dct.items():
        temp_df[method_key] = method_val

        data[method_key] = -1
        for idx in temp_df.index:
            data.loc[idx, method_key] = temp_df.loc[idx, method_key]

    return data


def plot_sub_plot(
        ax,
        row: int,
        column: int,
        data: pd.DataFrame,
        color_params: dict,
        lipid_dct: dict = None,
        max_y=-1,
):
    tmp_ax = ax[row, column]
    data.sort_values(by="lipid", ascending=True, inplace=True)
    emphasis_lipid_classes = ["CE", "ST"]
    # Calculate mean and standard deviation
    mean_line = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    # Add standard deviation to the mean line with alpha=0.6, ensuring it is on top
    x_labels = data.columns
    for i, r in data.iterrows():
        tmp_lipid_class = lipid_dct.get(i, "Others")
        if tmp_lipid_class not in emphasis_lipid_classes:
            tmp_color_cfg = color_params.get(tmp_lipid_class, ["#7E7E7E", ":"])
            tmp_color = tmp_color_cfg[0]
            tmp_linestyle = tmp_color_cfg[1]
            ax[row, column].plot(
                r,
                color=tmp_color,
                linestyle=tmp_linestyle,
                linewidth=3,
                alpha=0.33,
                label=tmp_lipid_class,
            )
            # ax[row, column].scatter(
            #     x_labels,
            #     r,
            #     alpha=0.5,
            #     marker="o",
            #     facecolors="none",
            #     edgecolors="#424242",
            #     linewidths=2,
            #     s=20,
            # )
    # plot emphasis lipid classes on top layers
    for i, r in data.iterrows():
        tmp_lipid_class = lipid_dct.get(i, "Others")
        if tmp_lipid_class in emphasis_lipid_classes:
            tmp_color_cfg = color_params.get(tmp_lipid_class, ["#7E7E7E", ":"])
            tmp_color = tmp_color_cfg[0]
            tmp_linestyle = tmp_color_cfg[1]
            ax[row, column].plot(
                r,
                color=tmp_color,
                linestyle=tmp_linestyle,
                linewidth=3,
                alpha=0.33,
                label=tmp_lipid_class,
            )
            # ax[row, column].scatter(
            #     x_labels,
            #     r,
            #     alpha=0.5,
            #     marker="o",
            #     facecolors="none",
            #     edgecolors="#424242",
            #     linewidths=2,
            #     s=20,
            # )

    # Plot mean line
    ax[row, column].fill_between(x_labels, mean_line - std_dev, mean_line + std_dev, color='#BABEC9', alpha=0.5,
                                 label='Standard Deviation', zorder=190)
    ax[row, column].plot(x_labels, mean_line, color='#2D3039', linewidth=6, label='Mean Trend', linestyle=":",
                         alpha=0.95, zorder=200)

    # # Plot mean line
    # ax[row, column].fill_between(x_labels, mean_line - std_dev, mean_line + std_dev, color='#FFADD1', alpha=0.36,
    #                              label='Standard Deviation', zorder=190)
    # ax[row, column].plot(x_labels, mean_line, color='#F5006A', linewidth=6, label='Mean Trend', linestyle="--",
    #                      alpha=0.95, zorder=200)

    if data.to_numpy().max() < 2:
        tmp_ax.set_ylim([-1.75, 1.75])
        tmp_ax.axhline(y=0, color="#424242", linestyle="--")
        tmp_ax.axhline(y=1, color="#9E9E9E", linestyle="--")
        tmp_ax.axhline(y=-1, color="#9E9E9E", linestyle="--")
        tmp_ax.axhline(y=0.5, color="#BFBFBF", linestyle="--")
        tmp_ax.axhline(y=-0.5, color="#BFBFBF", linestyle="--")
        tmp_ax.axhline(y=1.5, color="#BFBFBF", linestyle="--")
        tmp_ax.axhline(y=-1.5, color="#BFBFBF", linestyle="--")
    if max_y > 0:
        tmp_ax.set_ylim([0, max_y])
    for x_val in data.columns:
        tmp_ax.axvline(x=x_val, color="#9E9E9E", linestyle=":")

    handles, labels = tmp_ax.get_legend_handles_labels()
    unique_legends = [
        (th, tl)
        for i, (th, tl) in enumerate(zip(handles, labels))
        if tl not in labels[:i]
    ]
    return unique_legends


def plot_sub_lipid_class_plot(
        ax,
        row: int,
        column: int,
        data: pd.DataFrame,
        color_params: dict,
        lipid_dct: dict = None,
        lipid_classes: list = None,
        max_y=-1,
):
    tmp_ax = ax[row, column]
    data.sort_values(by="lipid", ascending=True, inplace=True)
    # for i, r in data.iterrows():
    #     tmp_lipid_class = lipid_dct.get(i, "Others")
    #     if tmp_lipid_class in lipid_classes:
    #         tmp_color_cfg = color_params.get(tmp_lipid_class, ["#7E7E7E", ":"])
    #         tmp_color = tmp_color_cfg[0]
    #         tmp_linestyle = tmp_color_cfg[1]
    #         ax[row, column].plot(
    #             r,
    #             color=tmp_color,
    #             linestyle=tmp_linestyle,
    #             linewidth=3,
    #             alpha=0.33,
    #             label=tmp_lipid_class,
    #         )
    emphasis_lipid_classes = ["CE", "ST"]
    plotted_lipids = 0
    # Calculate mean and standard deviation
    mean_line = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    # Add standard deviation to the mean line with alpha=0.6, ensuring it is on top
    x_labels = data.columns

    for i, r in data.iterrows():
        tmp_lipid_class = lipid_dct.get(i, "Others")
        if tmp_lipid_class in lipid_classes and tmp_lipid_class not in emphasis_lipid_classes:
            tmp_color_cfg = color_params.get(tmp_lipid_class, ["#7E7E7E", ":"])
            tmp_color = tmp_color_cfg[0]
            tmp_linestyle = tmp_color_cfg[1]
            ax[row, column].plot(
                r,
                color=tmp_color,
                linestyle=tmp_linestyle,
                linewidth=3,
                alpha=0.33,
                label=tmp_lipid_class,
            )
            # ax[row, column].scatter(
            #     x_labels,
            #     r,
            #     alpha=0.5,
            #     marker="o",
            #     facecolors="none",
            #     edgecolors="#424242",
            #     linewidths=2,
            #     s=20,
            # )
            plotted_lipids += 1
    # plot emphasis lipid classes on top layers
    for i, r in data.iterrows():
        tmp_lipid_class = lipid_dct.get(i, "Others")
        if tmp_lipid_class in lipid_classes and tmp_lipid_class in emphasis_lipid_classes:
            tmp_color_cfg = color_params.get(tmp_lipid_class, ["#7E7E7E", ":"])
            tmp_color = tmp_color_cfg[0]
            tmp_linestyle = tmp_color_cfg[1]
            ax[row, column].plot(
                r,
                color=tmp_color,
                linestyle=tmp_linestyle,
                linewidth=3,
                alpha=0.33,
                label=tmp_lipid_class,
            )
            # ax[row, column].scatter(
            #     x_labels,
            #     r,
            #     alpha=0.5,
            #     marker="o",
            #     facecolors="none",
            #     edgecolors="#424242",
            #     linewidths=2,
            #     s=20,
            # )
            plotted_lipids += 1

    # Plot mean line
    # ax[row, column].fill_between(x_labels, mean_line - std_dev, mean_line + std_dev, color='#FFADD1', alpha=0.2,
    #                              label='Standard Deviation', zorder=1)
    # ax[row, column].plot(x_labels, mean_line, color='#F5006A', linewidth=6, label='Mean Trend', linestyle="--",
    #                      alpha=0.95, zorder=200)

    # # Plot mean line
    ax[row, column].fill_between(x_labels, mean_line - std_dev, mean_line + std_dev, color='#BABEC9', alpha=0.4,
                                 label='Standard Deviation', zorder=1)
    ax[row, column].plot(x_labels, mean_line, color='#2D3039', linewidth=6, label='Mean Trend', linestyle=":",
                         alpha=0.95, zorder=200)

    if data.to_numpy().max() < 2:
        tmp_ax.set_ylim([-1.75, 1.75])
        tmp_ax.axhline(y=0, color="#424242", linestyle="--")
        tmp_ax.axhline(y=1, color="#9E9E9E", linestyle="--")
        tmp_ax.axhline(y=-1, color="#9E9E9E", linestyle="--")
        tmp_ax.axhline(y=0.5, color="#BFBFBF", linestyle="--")
        tmp_ax.axhline(y=-0.5, color="#BFBFBF", linestyle="--")
        tmp_ax.axhline(y=1.5, color="#BFBFBF", linestyle="--")
        tmp_ax.axhline(y=-1.5, color="#BFBFBF", linestyle="--")
    if max_y > 0:
        tmp_ax.set_ylim([0, max_y])
    for x_val in data.columns:
        tmp_ax.axvline(x=x_val, color="#9E9E9E", linestyle=":")

    handles, labels = tmp_ax.get_legend_handles_labels()
    unique_legends = [
        (th, tl)
        for i, (th, tl) in enumerate(zip(handles, labels))
        if tl not in labels[:i]
    ]
    return unique_legends, plotted_lipids


def plot_trend_lines(
        data: pd.DataFrame,
        save_path: str = "output.png",
        methods: list = None,
        color_cfg: str = "color_cfg.json",
        lipid_dct: dict = None,
        color_level="main_class_level",
):
    if methods is None:
        methods = ["hew", "km", "bisect_km", "gmm", "dpgmm"]

    color_params_dct = json.load(open(color_cfg, "r"))
    color_lv_params_dct = color_params_dct.get(color_level)

    max_rows = len(methods)
    max_cluster_lst = [data[f"{method}_cluster"].max() for method in methods]
    max_cluster_num = int(max(max_cluster_lst))
    print(f"Max cluster: {max_cluster_num + 1}")
    fig, ax = plt.subplots(
        nrows=max_rows,
        ncols=max_cluster_num + 1,
        # figsize=(max_cluster_num * 12, max_rows * 9),
        figsize=(max_cluster_num * 18, max_rows * 9),
    )
    df_drop_col_lst = [f"{method}_cluster" for method in methods]
    # df_drop_col_lst.extend(["var", "median"])
    all_legends = []
    for method in methods:
        row_idx = methods.index(method)
        for i in range(max_cluster_num + 1):
            cluster = data[data[f"{method}_cluster"] == i].copy()
            for col in df_drop_col_lst:
                if col in cluster.columns:
                    cluster = cluster.drop(columns=df_drop_col_lst)
            if not cluster.empty:
                tmp_unique_legends = plot_sub_plot(
                    ax, row_idx, i, cluster, color_lv_params_dct, lipid_dct
                )
                all_legends.extend(tmp_unique_legends)
                # print(f"plotting {method} cluster {i} at row {row_idx} col {i}")
            else:
                print(f"Empty plot of {method} cluster {i} @ row {row_idx} x col {i}")
            ax[row_idx, i].set_title(f"{method} Cluster {i}")
            del cluster

    color_labels = color_lv_params_dct.keys()

    all_labels = list([tal[1] for tal in all_legends])

    legend_handles = []
    legend_labels = []
    legend_label_colors = []
    for clb in color_labels:
        for th, tl in all_legends:
            if clb == tl:
                legend_handles.append(th)
                legend_labels.append(tl)
                legend_label_colors.append(color_lv_params_dct.get(tl)[0])
                break
            else:
                pass
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=math.ceil(len(color_lv_params_dct.keys()) / 3),
        labelcolor=legend_label_colors,
        fontsize=20,
    )
    fig.savefig(save_path)
    plt.close(fig)


def plot_sub_trend_lines(
        data: pd.DataFrame,
        save_path: str = "output.png",
        method: str = None,
        sub_groups: list = None,
        color_cfg: str = "color_cfg.json",
        lipid_dct: dict = None,
        color_level="main_class_level",
        sync_axis=False,
):
    if method is None:
        method = "gmm"

    if sub_groups is None:
        sub_groups = [
            ["TG"],
            ["CE", "ST"],
            ["PL", "O-PL", "P-PL"],
            ["LPL", "O-LPL", "P-LPL"],
            ["SM", "Cer", "HexCer", "DihydroCer", "DeoxyCer", "PythoCer"],
            ["SM"],
            ["Cer"],
            ["HexCer"],
            ["DihydroCer"],
            ["DeoxyCer"],
            ["PhytoCer"],
        ]

    color_params_dct = json.load(open(color_cfg, "r"))
    color_lv_params_dct = color_params_dct.get(color_level)

    max_rows = 1 + len(sub_groups)
    max_cluster_lst = [data[f"{method}_cluster"].max()]
    max_cluster_num = int(max(max_cluster_lst))
    print(f"Max cluster: {max_cluster_num + 1}")
    fig, ax = plt.subplots(
        nrows=max_rows,
        ncols=max_cluster_num + 1,
        # figsize=(max_cluster_num * 12, max_rows * 9),
        figsize=(max_cluster_num * 18, max_rows * 9),
    )
    df_drop_col_lst = [f"{method}_cluster"]
    # df_drop_col_lst.extend(["var", "median"])
    all_legends = []
    max_df_y = -1
    if sync_axis:
        max_data_df = data.copy()
        max_data_df = max_data_df.drop(columns=df_drop_col_lst)
        max_df_y = max_data_df.max().max() * 1.05
    for i in range(max_cluster_num + 1):
        cluster = data[data[f"{method}_cluster"] == i].copy()

        for col in df_drop_col_lst:
            if col in cluster.columns:
                cluster = cluster.drop(columns=col)
        if not cluster.empty:
            tmp_unique_legends = plot_sub_plot(
                ax, 0, i, cluster, color_lv_params_dct, lipid_dct, max_y=max_df_y
            )
            all_legends.extend(tmp_unique_legends)
            # print(f"plotting {method} cluster {i} at row {row_idx} col {i}")
        else:
            print(f"Empty plot of {method} cluster {i} @ row {0} x col {i}")
        ax[0, i].set_title(f"{method} Cluster {i}")
        del cluster
    for sub_group in sub_groups:
        row_idx = sub_groups.index(sub_group) + 1
        sub_lipids_names = []
        if sync_axis:
            for l_n in lipid_dct:
                l_c = lipid_dct.get(l_n, "Others")
                if l_c in sub_group:
                    sub_lipids_names.append(l_n)
            sub_l_df = data[data.index.isin(sub_lipids_names)]
            df_drop_col_lst = [f"{method}_cluster"]
            # df_drop_col_lst.extend(["var", "median"])
            sub_l_df = sub_l_df.drop(columns=df_drop_col_lst)
            sub_group_max_y = sub_l_df.max().max() * 1.05
        else:
            sub_group_max_y = -1
        for i in range(max_cluster_num + 1):
            cluster = data[data[f"{method}_cluster"] == i].copy()
            for col in df_drop_col_lst:
                if col in cluster.columns:
                    cluster = cluster.drop(columns=df_drop_col_lst)
            if not cluster.empty:
                tmp_unique_legends, plotted_lipids_count = plot_sub_lipid_class_plot(
                    ax,
                    row_idx,
                    i,
                    cluster,
                    color_lv_params_dct,
                    lipid_dct,
                    sub_group,
                    max_y=sub_group_max_y,
                )
                all_legends.extend(tmp_unique_legends)
                # print(f"plotting {method} cluster {i} at row {row_idx} col {i}")
            else:
                print(f"Empty plot of {method} cluster {i} @ row {row_idx} x col {i}")
                plotted_lipids_count = 0
            sub_group_str = " + ".join(sub_group)
            ax[row_idx, i].set_title(f"{method} Cluster {i}: {sub_group_str} #lipid count: {plotted_lipids_count}")
            del cluster

    color_labels = color_lv_params_dct.keys()

    all_labels = list([tal[1] for tal in all_legends])

    legend_handles = []
    legend_labels = []
    legend_label_colors = []
    for clb in color_labels:
        for th, tl in all_legends:
            if clb == tl:
                legend_handles.append(th)
                legend_labels.append(tl)
                legend_label_colors.append(color_lv_params_dct.get(tl)[0])
                break
            else:
                pass
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=math.ceil(len(color_lv_params_dct.keys()) / 3),
        labelcolor=legend_label_colors,
        fontsize=20,
    )
    fig.savefig(save_path)
    fig.savefig(f'{save_path}.svg')
    plt.close(fig)


def add_cluster_to_raw_data(
        raw_data: pd.DataFrame,
        cluster_data: pd.DataFrame,
        method: str = 'gmm',
):
    if method is None:
        methods = "gmm"

    cluster_methods_lst = [f"{method}_cluster"]
    cluster_df = cluster_data[cluster_methods_lst].copy()
    raw_data = pd.concat([raw_data, cluster_df], axis=1)
    return raw_data
