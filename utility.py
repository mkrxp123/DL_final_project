import os
import json
from easydict import EasyDict
import torch
from torch import optim

def fetch_config(args):
    config = json.load(open(args.config,'r'))
    config = EasyDict(config)
    for k, v in vars(args).items():
        config[k] = v
    # config['device'] = args.device
    # config['config'] = args.config
    # config['validation'] = args.validation
    # config['name'] = args.name
    # config['ckpt'] = args.ckpt
    # config['batch_size'] = args.batch_size
        
    return config

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    if args.scheduler == 'Cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(85))
    elif args.scheduler == 'Reduce':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, min_lr = 0.000005, patience=0,  eps=args.epsilon)
    elif args.scheduler == 'OneCycle':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    elif args.scheduler == 'MultiCycle':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.000005, max_lr=args.lr, # 9000  171000
            step_size_up=9000, step_size_down=171000, mode='triangular2',cycle_momentum = False)  #mode in ['triangular', 'triangular2', 'exp_range']
    return optimizer, scheduler

def load_model(checkpoint_path, model):
    model_dict = model.state_dict()    

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_med = checkpoint['model']

    for k, v in model_med.items():
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict:
            model_dict[k].copy_(v)
        else:
            print('Warning: key %s not found in model' % k)
    model.load_state_dict(model_dict, strict=True)
    model.eval()
    # device = torch.device('cuda:'+ str(args.gpuid[0]))
    # model.to(device) 
    # print(checkpoint_path)