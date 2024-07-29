import argparse

import os
import torch

#dir_path = os.path.dirname(os.path.realpath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Parameters for Contrastive Predictive Coding for Human '
                    'Activity Recognition')

    # Data loading parameters
    #parser.add_argument('--window', type=int, default=50, help='Window size')
    '''parser.add_argument('--overlap', type=int, default=25,
                        help='Overlap between consecutive windows')
    '''
    
    parser.add_argument('--learning_rate', type=float, default=5e-4)

    parser.add_argument('--num_epochs', type=int, default=70)
    parser.add_argument('--gpu_id', type=str, default='0')

    # Dataset to train on
    parser.add_argument('--dataset', type=str, 
                        default='MESA',
                        help='Choosing the dataset to perform the training on')

    # Conv encoder
    parser.add_argument('--kernel_size', type=int, default=5,
                        help='Size of the conv filters in the encoder')

    # Future prediction horizon
    '''parser.add_argument('--num_steps_prediction', type=int, default=12,
                        help='Number of steps in the future to predict')
    '''
    # ------------------------------------------------------------
    # Classification parameters
    #parser.add_argument('--classifier_lr', type=float, default=0.001)
    parser.add_argument('--classifier_lr', type=float, default=5e-4)
    
    parser.add_argument('--learning_schedule', type=str, default='all_layers',
                        choices=['last_layer', 'all_layers'],
                        help='last layer freezes the encoder weights but '
                             'all_layers does not.')
    # ------------------------------------------------------------

    # Random seed for reproducibility
    parser.add_argument('--random_seed', type=int, default=42)

    args = parser.parse_args()

    # Setting parameters by the dataset
    if args.dataset == 'MIMIC':
        args.input_size = 76
        args.window = 48
        args.num_steps_prediction =28
        args.data_file = '../SimCLR/MIMIC'
        args.saved_model = 'models/Dec-22-2023/MIMIC_k_28_lr_0.0005_bs_64.pkl'
        args.batch_size = 64
        args.classifier_batch_size = 128

    if args.dataset == 'GLOBEM':
        args.input_size = 1390
        args.window = 28
        args.num_steps_prediction = 12
        args.data_file = '../SimCLR/GLOBEM'
        args.saved_model = 'models/Nov-24-2023/GLOBEM_k_12_lr_0.0005_bs_64.pkl'
        args.batch_size = 64
        args.classifier_batch_size = 128

    if args.dataset == 'MESA':
        args.input_size = 5
        args.window = 101
        args.num_steps_prediction = 28
        args.data_file = '../SimCLR/MESA'
        args.saved_model = 'models/Dec-19-2023/MESA_k_28_lr_0.0005_bs_512.pkl'
        args.batch_size = 512
        args.classifier_batch_size = 512

    args.num_classes = 2
    args.device = torch.device(
        "cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    
    # Conv padding size
    args.padding = int(args.kernel_size // 2)

    return args

