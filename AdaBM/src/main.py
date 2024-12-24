import torch
import torch.nn as nn
import utility
import data
import model
from option import args
from trainer import Trainer

import time
import datetime

import numpy
import random

# 设置PyTorch手动种子，以确保实验的可重复性
torch.manual_seed(args.seed)
# 设置CuDNN确定性模式，以确保卷积操作在不同平台上表现一致
torch.backends.cudnn.deterministic=True
# 关闭CuDNN基准测试模式，以确保每次运行的结果一致
torch.backends.cudnn.benchmark=False
# 设置NumPy随机种子，进一步确保整个实验环境的确定性
numpy.random.seed(args.seed)
# 设置Python内置随机模块种子，覆盖所有随机操作
random.seed(args.seed)
# 设置PyTorch CUDA手动种子，针对GPU操作确保可重复性
torch.cuda.manual_seed(args.seed)
# 设置所有GPU的PyTorch CUDA手动种子，用于多GPU环境
torch.cuda.manual_seed_all(args.seed)
# 初始化检查点，用于模型训练过程中保存和加载模型
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    exp_start_time = time.time()
    _loader = data.Data(args)
    _model = model.Model(args, checkpoint)
    t = Trainer(args, _loader, _model, checkpoint)
    
    # t.test_teacher()
    while not t.terminate():
        torch.manual_seed(args.seed)
        t.train()   
        t.test()

    exp_end_time = time.time()
    exp_time_interval = exp_end_time - exp_start_time
    t_string = "Total Running Time is: " + str(datetime.timedelta(seconds=exp_time_interval)) + "\n"
    checkpoint.write_log('{}'.format(t_string))
    checkpoint.done()