import torch
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert spconv2 pth')
    parser.add_argument('pth', help='input pth')

    args = parser.parse_args()

    input_pth = args.pth
    backup_pth = input_pth+'.bak'

    saved = torch.load(input_pth)
    torch.save(saved, backup_pth)

    for k in saved['state_dict']:
        if 'backbone' in k and k.endswith('weight') and 'ema_pts_segmentor__backbone' in k:
            print('converting', k)
            param = saved['state_dict'][k]
            if len(param.shape) > 2:
                # dims = list(range(1, len(param.shape))) + [0]
                dims = [4,0,1,2,3]
                param = param.permute(*dims)
            saved['state_dict'][k] = param

    torch.save(saved, input_pth)