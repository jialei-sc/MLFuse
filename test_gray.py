import torch
import argparse
from network import TransNet
from utils import test
from collections import OrderedDict
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='model save and load')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--model_path', type=str, default='./model/model_fuse.pth', help='best model path')
parser.add_argument('--test_path', type=str, default='./test_dataset/ct_mri_set/', help='test dataset path')
parser.add_argument('--result_path', type=str, default='./test_result/test_ct/', help='test result path')
args = parser.parse_args()
device = 'cuda'


def print_model_param_nums(model=None):
    if model == None:
        model = torchvision.models.alexnet()
    total = sum([param.nelement() if param.requires_grad else 0 for param in model.parameters()])
    print('  + Number of params: %.4fM' % (total / 1e6))

    
model = TransNet().to(device)
print_model_param_nums(model)


state_dict = torch.load(args.model_path, map_location='cuda:0')['model']


if len(args.gpus) > 1:
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
else:
    model.load_state_dict(state_dict)

model.eval()

test(args.test_path, model, save_path=args.result_path)

# print("total fusion time：", total_time, " used")