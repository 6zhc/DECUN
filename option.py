import argparse
import os
import warnings
import torch

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default='./trained_models/')
parser.add_argument('--result_folder', type=str, default='result')
parser.add_argument('--test_folder', type=str, default='data')
parser.add_argument('--model', type=str, default='DecovNetConvergence')
parser.add_argument('--trained_model', type=str, default='DecovNetConvergence_4_30.pk')

parser.add_argument('--filter_num', type=int, default=4)
parser.add_argument('--layer_num', type=int, default=30)
parser.add_argument('--convergence_type', type=int, default=0)
parser.add_argument('--noise', type=float, default=0.01)

parser.add_argument('--model_option', type=str, default='')
parser.add_argument('--bs', type=int, default=8, help='batch size')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--eval_step', type=int, default=30)
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--random_initial', action='store_true')


opt = parser.parse_args()
opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(opt)
