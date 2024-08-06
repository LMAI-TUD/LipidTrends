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
import numpy as np
import pandas as pd
from scipy import stats


def get_normalized_data(data, mode="zscore"):
    scaled_df = data.copy()

    # zscore
    ds_file_np = scaled_df.to_numpy()
    scaled_df = pd.DataFrame(
        data=ds_file_np, index=scaled_df.index, columns=list(scaled_df)
    )
    if mode == "zscore":
        scaled_df = scaled_df.apply(stats.zscore, 1, False, "broadcast")
    elif mode == "autoscale":
        scaled_df = scaled_df.apply(stats.zscore, 1, False, "broadcast")
    elif mode == "log2":
        scaled_df = scaled_df.apply(lambda x: np.log2(x + 1), 1, False, "broadcast")
    elif mode == "min_max":
        scaled_df = scaled_df.apply(
            lambda x: (x - x.mean()) / x.std(), 1, False, "broadcast"
        )
    elif mode == "pareto":
        scaled_df = scaled_df.apply(
            lambda x: (x - x.mean()) / np.sqrt(x.std()), 1, False, "broadcast"
        )
    else:
        # set Z-score as default
        scaled_df = scaled_df.apply(stats.zscore, 1, False, "broadcast")

    return scaled_df


def preprocess_file(input_file, lipid_col=0, lipid_class_col=1):
    raw_df = pd.read_csv(input_file, sep=",")
    raw_df.rename(columns={raw_df.columns[lipid_col]: "lipid"}, inplace=True)
    lipid_class_col_name = raw_df.columns[lipid_class_col]
    lipid_df = pd.DataFrame(raw_df, columns=["lipid", lipid_class_col_name])
    lipid_df.drop_duplicates(inplace=True)
    lipid_df.set_index("lipid", inplace=True)
    lipid_dct = lipid_df.to_dict().get(lipid_class_col_name, {})
    raw_df.drop(columns=[lipid_class_col_name], axis=1, inplace=True)

    unique_lipid_names = raw_df["lipid"].unique().tolist()
    df = raw_df.groupby("lipid").sum()
    # df = raw_df

    for idx, row in df.iterrows():
        row_non_zero = row[row != 0]
        row_non_zero_min = row_non_zero.min()
        for col in df.columns:
            if row[col] == 0:
                df.loc[idx, col] = row_non_zero.min() / 2
                print(f"replaced {idx} {col} with min /2 to {row_non_zero_min / 2}")

    variance = df.var(axis=1)
    median = df.median(axis=1)

    # remove duplicated indices
    # df = df.loc[~df.index.duplicated(keep="first")]
    return df, lipid_dct
