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
import os

import numpy as np
import pandas as pd


def load_file(file, na_values: list = None, index_col=0, header_row=0, zero_is_na: bool = True):
    if isinstance(na_values, list) and na_values:
        pass
    elif isinstance(na_values, str) and na_values:
        na_values = [na_values]
    else:
        na_values = ["NA", "na", "N/A", "n/a", "NaN", "nan", "NAN"]

    if zero_is_na:
        na_values.append(0)

    if file.endswith(".csv"):
        df = pd.read_csv(file, na_values=na_values, index_col=index_col, header=header_row)
    elif file.endswith(".xlsx"):
        df = pd.read_excel(file, na_values=na_values, index_col=index_col, header=header_row)
    else:
        raise ValueError("File must be .csv or .xlsx")
    # check if index is unique
    if df.index.duplicated().any():
        raise ValueError("Index/row names must be unique, check the input file for duplicated lipids/sample names")
    if df.columns.duplicated().any():
        raise ValueError("Columns must be unique, check the input file for duplicated lipids/sample names")
    return df


def save_file(df, file, index=True):
    if file.endswith(".csv"):
        df.to_csv(file, index=index)
    elif file.endswith(".xlsx"):
        df.to_excel(file, index=index)
    else:
        raise ValueError("File must be .csv or .xlsx")

    abs_output_path = None
    if os.path.isfile(file):
        abs_output_path = os.path.abspath(file)

    return abs_output_path


def replace_min(df, axis=0, min_value_ratio=5):
    df_fill = df.copy()

    if min_value_ratio < 1:
        min_value_ratio = 5

    df_replace_lst = []

    if axis == 0:
        pass
    elif axis == 1:
        df_fill = df_fill.T
    else:
        raise ValueError("axis must be 0 or 1")
    col_names = df.columns.tolist()
    for i, r in df.iterrows():
        raw_min = r.min()
        min_i = r.min() / min_value_ratio
        # print(f"row {i} raw min: {raw_min} , 1/5 min: {min_i}")
        for col in col_names:
            if np.isnan(r[col]):
                df_fill.at[i, col] = min_i
                print(f"! Missing value detected: row {i} column {col} has N/A value: {df.loc[i, col]}.\n"
                      f"> Fill this cell with {min_i}.  "
                      f"# 1/{min_value_ratio} of the min value in this row {raw_min}.\n")

    # transpose back
    if axis == 0:
        pass
    elif axis == 1:
        df_fill = df_fill.T
    else:
        raise ValueError("axis must be 0 or 1")
    return df_fill


def replace_min_file(
        file,
        output_file=None,
        na_values: list = None,
        zero_is_na: bool = False,
        min_value_ratio=5,
        index_col=0,
):
    if isinstance(na_values, list) and na_values:
        pass
    elif isinstance(na_values, str) and na_values:
        na_values = [na_values]
    else:
        na_values = ["NA", "na", "N/A", "n/a", "NaN", "nan", "NAN"]

    if zero_is_na:
        na_values.append(0)

    df = load_file(file, na_values=na_values, index_col=index_col)
    df_fill = replace_min(df, min_value_ratio=min_value_ratio)

    if output_file:
        output_path = save_file(df_fill, output_file)
        print(f"Saved file to: {output_path}")

    return df_fill


def average_group(df, meta, group_col, sample_col, keep_original=False):
    meta_groups = meta[group_col].unique()

    group_sample_dct = {}
    for i, r in meta.iterrows():
        if r[group_col] in meta_groups:
            if r[group_col] in group_sample_dct:
                group_sample_dct[r[group_col]].append(r[sample_col])
            else:
                group_sample_dct[r[group_col]] = [r[sample_col]]

    df_avg = df.copy()
    for gp in group_sample_dct:
        df_avg[gp] = df[group_sample_dct[gp]].mean(axis='columns')

    if not keep_original:
        original_cols = df.columns.tolist()
        df_avg.drop(original_cols, axis=1, inplace=True)

    return df_avg


def calc_avg(
        df,
        meta,
        # output_file,
        group_col,
        sample_col,
        axis=0,
        keep_original=True,
):
    df_avg = average_group(
        df, meta, group_col=group_col, sample_col=sample_col, keep_original=keep_original
    )
    # output_path = save_file(df_avg, output_file)
    return df_avg


def get_lipid_class_info(df, lipid_class_col):
    lipid_class_dct = dict(zip(df.index, df[lipid_class_col]))
    return lipid_class_dct
