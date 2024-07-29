import os
import pandas as pd
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Evaluation with bootstrapping
np.random.seed(1234)
rng = np.random.RandomState(1234)

from PIL import Image, ImageDraw, ImageFont


def get_paths(dataset):
    '''returns list of file paths for all models for the specified dataset'''

    if dataset == 'MIMIC':
        paths = [
            os.path.join('Supervised', dataset, 'l2_e100_esFalse_bs128_wTrue', 'supervised.finetuned_fairness_metrics.csv'),   
            os.path.join('CNN AE', dataset, 'hs128_e200_esTrue_bs128_wTrue_rFalse', 'cnn_ae.finetuned_fairness_metrics.csv'),
            os.path.join('SimCLR', dataset, 'e100_esFalse_bs128_wTrue_f1_m', 'simclr.frozen_fairness_metrics.csv'),
            os.path.join('CPC', dataset, 'CPC_fairness_metrics.csv')
        ]

    if dataset == 'MESA':
        paths = [
            os.path.join('Supervised', dataset, 'l2_e100_esFalse_bs128_wTrue', 'supervised.finetuned_fairness_metrics.csv'),
            os.path.join('CNN AE', dataset, 'hs128_e200_esTrue_bs1024_wTrue_rFalse', 'cnn_ae.finetuned_fairness_metrics.csv'),
            os.path.join('SimCLR', dataset, 'e100_esFalse_bs128_wTrue_f1_m', 'simclr.frozen_fairness_metrics.csv'),
            os.path.join('CPC', dataset, 'CPC_fairness_metrics.csv')
        ]

    if dataset == 'GLOBEM':
        paths = [
            os.path.join('Supervised', dataset, 'l2_e200_esTrue_bs64_wTrue', 'supervised.finetuned_fairness_metrics.csv'),
            os.path.join('CNN AE', dataset, 'hs128_e200_esTrue_bs128_wTrue_rFalse', 'cnn_ae.finetuned_fairness_metrics.csv'),
            os.path.join('SimCLR', dataset, 'e100_esFalse_bs128_wTrue_f1_m', 'simclr.frozen_fairness_metrics.csv'),
            os.path.join('CPC', dataset, 'CPC_fairness_metrics.csv')

        ]

    return paths


def metrics_per_dataset(results_folder, dataset):
    '''returns a df of fairness metrics for all models & the specified dataset'''

    fairness_metrics = pd.DataFrame()
    paths = get_paths(dataset)
    for path in paths:
        fairness_tmp = pd.read_csv(path)
        selected_columns = ['protected_attribute','fairness_metric', 'value', 'tag']
        fairness_tmp = fairness_tmp[selected_columns]
        fairness_metrics = pd.concat([fairness_metrics, fairness_tmp], axis=0)
    #print(fairness_metrics)

    #fairness_metrics.to_csv(os.path.join(results_folder, 'fairness_metrics.csv'), index=False)

    return fairness_metrics


def split_df(dataset, fairness_metrics, results_folder):
    '''split metrics df to ratios & differences for the specified dataset
    and add column Parity Deviation'''

    if dataset == 'MESA':
        ratios = ['false_positive_rate_ratio', 'error_rate_ratio']
        df_with_ratios = fairness_metrics[fairness_metrics['fairness_metric'].isin(ratios)]
        diff = ['average_absolute_odds_difference']
        df_with_diff = fairness_metrics[fairness_metrics['fairness_metric'].isin(diff)]
    else:
        ratios = ['false_negative_rate_ratio', 'error_rate_ratio']
        df_with_ratios = fairness_metrics[fairness_metrics['fairness_metric'].isin(ratios)]
        diff = ['average_absolute_odds_difference']
        df_with_diff = fairness_metrics[fairness_metrics['fairness_metric'].isin(diff)]    
        
    # Calculate deviation from fairness parity (1.0 for ratios)  
    df_with_ratios.loc[:, "value"].astype(float)
    df_with_ratios.loc[:, "Parity Deviation"] = abs(1-df_with_ratios.value)
    df_with_ratios

    df_with_ratios.tag.value_counts()

    tag_mapping = {'supervised': '1D CONV', 'cnn': '1D CONV AE', 'simclr': 'SimCLR', 'cpc': 'CPC'}
    df_with_ratios['tag'] = df_with_ratios['tag'].replace(tag_mapping)
    #print(df_with_ratios)

    # Calculate deviation from fairness parity (0.0 for differences)  
    df_with_diff.loc[:, "value"].astype(float)
    df_with_diff.loc[:, "Parity Deviation"] = abs(df_with_diff.value)
    df_with_diff

    df_with_diff.tag.value_counts()

    df_with_diff['tag'] = df_with_diff['tag'].replace(tag_mapping)
    print(df_with_diff)

    max_parity_deviation_ratio = df_with_ratios['Parity Deviation'].max()
    print(f"Maximum Parity Deviation for ratios: {max_parity_deviation_ratio}")  

    max_parity_deviation_diff = df_with_diff['Parity Deviation'].max()
    print(f"Maximum Parity Deviation for differences: {max_parity_deviation_diff}")

    median_parity_deviation_diff = df_with_diff.groupby('tag')['Parity Deviation'].median()
    print(median_parity_deviation_diff)

    df_with_ratios.to_csv(os.path.join(results_folder, 'ratios_fairness_metrics.csv'), index=False)

    return df_with_ratios, df_with_diff



def boxplot_and_medians(results_folder, df_with_ratios, all_models):
    '''Creates a boxplot of Parity Deviation (ratios) for each model & a df
    for medians and IQR for each model'''
     
    plt.clf()
    sns.set(style="whitegrid")
    sns.set_context('poster')
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='tag', y='Parity Deviation', hue='tag', data=df_with_ratios, dodge=False)
    plt.axhline(y=0.2, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Model')
    plt.ylabel('Parity Deviation')
    #plt.xticks(fontsize=14)
    plt.legend([],[], frameon=False)
    #plt.title('Boxplots of Parity Deviation by Model')
    if all_models==False:
        plt.savefig(os.path.join(results_folder, 'ratios_boxplot.png'),dpi=600, bbox_inches='tight')
    else:
        plt.savefig("results/boxplot_all_datasets.png", dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

    #get medians & IQRs for ratios & each model
    medians_iqrs = df_with_ratios.groupby('tag')['Parity Deviation'].agg(['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
    medians_iqrs.columns = ['median', 'Q1', 'Q3']
    medians_iqrs['IQR'] = medians_iqrs['Q3'] - medians_iqrs['Q1']
    medians_iqrs = medians_iqrs.reset_index()
    print(medians_iqrs)
    if all_models==False:
        medians_iqrs.to_csv(os.path.join(os.path.join(results_folder, 'medians_iqrs.csv')), index=False)
    else:
        medians_iqrs.to_csv('results/all_datasets_medians_iqrs.csv', index=False)





def diff_barplot(results_folder, df_with_diff, all_models=False):
    '''Creates a bar plot of medians for Parity Deviation (differences for each model'''

    plt.clf()
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Create a bar plot of medians for Parity Deviation for each model
    sns.barplot(x='tag', y='Parity Deviation', data=df_with_diff, estimator='median', ci=None, color='navy', width=0.3)
    plt.axhline(y=0.1, color='red', linestyle='--', linewidth=2, label='Threshold')
    plt.xlabel('Model')
    plt.ylabel('Parity Deviation (Median)')
    #plt.title('Median Parity Deviation for Each Model')
    if all_models==False:
        plt.savefig(os.path.join(results_folder, 'differences_barplot.png'))
    else:
        plt.savefig("results/barplot_all_datasets.png",bbox_inches='tight', dpi=300)
    plt.show()




def merge_images(image1_path, image2_path, image3_path, output_path):
    '''merges barplots and boxplots'''

    image1 = plt.imread(image1_path)
    image2 = plt.imread(image2_path)
    image3 = plt.imread(image3_path)

    fig, axs = plt.subplots(3, 1, figsize=(5, 15))

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    axs[0].imshow(image1)
    axs[1].imshow(image2)
    axs[2].imshow(image3)

    # Add labels for each subplot
    axs[0].set_title("GLOBEM", fontsize=4)
    axs[1].set_title("MIMIC", fontsize=4)
    axs[2].set_title("MESA", fontsize=4)

    # Adjust layout to reduce empty space between subplots
    plt.subplots_adjust(hspace=-0.4)

    plt.savefig(output_path, bbox_inches='tight',dpi=600)


def merge_images_horiz(image1_path, image2_path, image3_path, output_path):
    '''merges barplots and boxplots'''

    image1 = plt.imread(image1_path)
    image2 = plt.imread(image2_path)
    image3 = plt.imread(image3_path)

    fig, axs = plt.subplots(1, 3, figsize=(5, 15))

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    axs[0].imshow(image1)
    axs[1].imshow(image2)
    axs[2].imshow(image3)

    # Add labels for each subplot
    axs[0].set_title("GLOBEM", fontsize=4)
    axs[1].set_title("MIMIC", fontsize=4)
    axs[2].set_title("MESA", fontsize=4)

    # Adjust layout to reduce empty space between subplots
    plt.subplots_adjust(wspace=0.02)

    plt.savefig(output_path, bbox_inches='tight',dpi=600)



datasets =['GLOBEM', 'MIMIC', 'MESA']
for dataset in datasets:
    results_folder = os.path.join('results', dataset)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    fairness_metrics = metrics_per_dataset(results_folder, dataset)
    df_with_ratios, df_with_diff = split_df(dataset, fairness_metrics, results_folder)
    boxplot_and_medians(results_folder, df_with_ratios, all_models=False)
    #diff_barplot(results_folder, df_with_diff, all_models=False)


image1_path = os.path.join('results', 'GLOBEM', 'ratios_boxplot.png')
image2_path = os.path.join('results', 'MIMIC', 'ratios_boxplot.png')
image3_path = os.path.join('results', 'MESA', 'ratios_boxplot.png')
output_path = "results/all_boxplots.png"
merge_images(image1_path, image2_path, image3_path, output_path)
output_path = "results/all_boxplots_horiz.png"
merge_images_horiz(image1_path, image2_path, image3_path, output_path)


image1_path = os.path.join('results', 'GLOBEM', 'differences_barplot.png')
image2_path = os.path.join('results', 'MIMIC', 'differences_barplot.png')
image3_path = os.path.join('results', 'MESA', 'differences_barplot.png')
output_path = "results/all_barlots.png"
merge_images(image1_path, image2_path, image3_path, output_path)


image1_path = "results/auc_curve_globem.png"
image2_path = "results/auc_curve_mimic.png"
image3_path = "results/auc_curve_mesa.png"
output_path = "results/all_auc_curves.png"
merge_images(image1_path, image2_path, image3_path, output_path)
output_path = "results/all_auc_curves_horiz.png"
merge_images_horiz(image1_path, image2_path, image3_path, output_path)



#merge all datasets in one
datasets =['GLOBEM', 'MIMIC', 'MESA']

ratios_all_datasets = pd.DataFrame()
diff_all_datasets = pd.DataFrame()

for dataset in datasets:
    results_folder = os.path.join('results', dataset)
    fairness_tmp = pd.read_csv(os.path.join(results_folder, 'fairness_metrics.csv'))
    df_with_ratios, df_with_diff = split_df(dataset, fairness_tmp, results_folder)
    df_with_ratios['Dataset'] = dataset
    df_with_diff['Dataset'] = dataset
    ratios_all_datasets = pd.concat([ratios_all_datasets, df_with_ratios], axis=0)
    diff_all_datasets = pd.concat([diff_all_datasets, df_with_diff], axis=0)
print(diff_all_datasets)
unique_values = diff_all_datasets['Dataset'].unique()
print(unique_values)
unique_values = ratios_all_datasets['Dataset'].unique()
print(unique_values)

boxplot_and_medians(results_folder, ratios_all_datasets, all_models=True)
#diff_barplot(results_folder, diff_all_datasets, all_models=True)

