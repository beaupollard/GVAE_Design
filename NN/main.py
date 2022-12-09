from DVAE import VAE
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import utils as util
import copy
import time
import torch



BS=64
percent_train=0.8
d1=util.create_dataset()
# d1=torch.load('data_2.pt')#smd.run_sim(run_nums=2,out_data=2,num_repeats=1)
train=torch.utils.data.DataLoader(d1,batch_size=BS, shuffle=True)

model=VAE()

for i in range(1000):
    loss=model.training_step(train)

    print(i, loss)
torch.save(model.state_dict(), 'current_model7')