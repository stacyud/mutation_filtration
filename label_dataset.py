#!/usr/bin/env python

import argparse
import pandas as pd
import numpy as np
import xgboost as xgb


def feature_columns_check(df, feature_columns):
    for feature in feature_columns:
        if feature not in df.columns:
            df[feature] = 0
    return df


def main(input_tsv, model_path, output_tsv, threshold):

    technical_features = [
        't_depth_LoFreq', 't_alt_count_LoFreq', 't_ref_count_LoFreq',
        'Tumor_VAF_LoFreq', 'variant_qual_LoFreq',
        't_depth_Mutect', 't_alt_count_Mutect', 't_ref_count_Mutect',
        'Tumor_VAF_Mutect',
        'FILTER_Mutect_feature', 'FILTER_LoFreq_feature',
        'base_qual', 'clustered_events', 'germline',
        'haplotype', 'map_qual', 'multiallelic',
        'orientation', 'position', 'strand_bias', 'weak_evidence'
    ]

    annotation_features = [
        'Variant_Classification_feature',
        'Reference_Allele_feature',
        'Tumor_Seq_Allele2_feature',
        '5prime_feature', '3prime_feature',
        'gnomADg_AF', 'gnomADg_AF_raw', 'gnomADg_AF_popmax',
        'gnomADg_AF_afr', 'gnomADg_AF_amr', 'gnomADg_AF_eas',
        'gnomADg_AF_fin', 'gnomADg_AF_nfe', 'gnomADg_AF_asj',
        'gnomADg_AF_sas', 'gnomADg_AF_oth'
    ]

    features = technical_features + annotation_features

    df = pd.read_csv(input_tsv, sep="\t")
    df = feature_columns_check(df, features)

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    y_proba = model.predict_proba(df[features].fillna(0))[:, 1]

    df['Probability_score'] = y_proba
    df['Label'] = np.where(df['Probability_score'] > threshold, 'PASS', '')

    df.to_csv(output_tsv, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Prepared feature TSV")
    parser.add_argument("--model", required=True, help="XGBoost model JSON")
    parser.add_argument("--output", required=True, help="Labeled output TSV")
    parser.add_argument("--threshold", type=float, default=0.3)

    args = parser.parse_args()
    main(args.input, args.model, args.output, args.threshold)
