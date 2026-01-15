#!/usr/bin/env python

import argparse
import pandas as pd
import numpy as np


# ---------------------------
# Utility functions
# ---------------------------

def expand_filter_column(df, filter_col="FILTER", sep=";"):
    filters_split = df[filter_col].str.split(sep)

    unique_filters = sorted(set(
        f for sublist in filters_split.dropna() for f in sublist if f != "PASS"
    ))

    for f in unique_filters:
        df[f] = filters_split.apply(
            lambda x: int(f in x) if isinstance(x, list) else 0
        )

    return df


def feature_columns_check(df, feature_columns):
    for feature in feature_columns:
        if feature not in df.columns:
            df[feature] = 0
    return df


# ---------------------------
# Main
# ---------------------------

def main(input_tsv, output_tsv):

    vc_dict = {
        'Missense_Mutation': 0,
        'Silent': 1,
        'Intron': 2,
        'Nonsense_Mutation': 3,
        "3'UTR": 4,
        "5'Flank": 5,
        'Splice_Region': 6,
        "5'UTR": 7,
        'Splice_Site': 8,
        "3'Flank": 9,
        'IGR': 10,
        'Translation_Start_Site': 11,
        'Targeted_Region': 12,
        'RNA': 13,
        'In_Frame_Ins': 14,
        'Nonstop_Mutation': 15
    }

    nucl_dict = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

    df = pd.read_csv(input_tsv, sep="\t")

    # flanking bases
    df['5prime'] = df['flanking_bps'].str.split('', expand=True)[1]
    df['3prime'] = df['flanking_bps'].str.split('', expand=True)[3]

    # categorical encoding
    df['Variant_Classification_feature'] = df['Variant_Classification'].map(vc_dict)
    df['Reference_Allele_feature'] = df['Reference_Allele'].map(nucl_dict)
    df['Tumor_Seq_Allele2_feature'] = df['Tumor_Seq_Allele2'].map(nucl_dict)
    df['5prime_feature'] = df['5prime'].map(nucl_dict)
    df['3prime_feature'] = df['3prime'].map(nucl_dict)

    # quality fields
    df['variant_qual_LoFreq'] = (
        df['variant_qual_LoFreq'].replace({'.': 0.0}).fillna(0.0).astype(float)
    )
    df['variant_qual_Mutect'] = (
        df['variant_qual_Mutect'].replace({'.': 0.0}).fillna(0.0).astype(float)
    )

    # gnomAD AFs
    gnomad_cols = [
        'gnomADg_AF', 'gnomADg_AF_raw', 'gnomADg_AF_popmax',
        'gnomADg_AF_afr', 'gnomADg_AF_amr', 'gnomADg_AF_eas',
        'gnomADg_AF_fin', 'gnomADg_AF_nfe', 'gnomADg_AF_asj',
        'gnomADg_AF_sas', 'gnomADg_AF_oth'
    ]

    for col in gnomad_cols:
        df[col] = df[col].replace({'.': 0}).fillna(-1).astype(float)

    # FILTER features
    df['FILTER_Mutect_feature'] = (df['FILTER_Mutect'] == 'PASS').astype(int)
    df = expand_filter_column(df, filter_col="FILTER_Mutect", sep=";")

    df['FILTER_LoFreq_feature'] = (df['FILTER_LoFreq'] == 'PASS').astype(int)

    df.to_csv(output_tsv, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input TSV file")
    parser.add_argument("--output", required=True, help="Output prepared TSV file")

    args = parser.parse_args()
    main(args.input, args.output)
