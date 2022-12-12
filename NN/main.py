from DVAE import VAE
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import utils as util
import copy
import time
import torch



BS=32
percent_train=0.8
d1=util.create_dataset()
# d1=torch.load('data_2.pt')#smd.run_sim(run_nums=2,out_data=2,num_repeats=1)
train=torch.utils.data.DataLoader(d1,batch_size=BS, shuffle=True)

correct_bodies=[]
miss_identification_bodies=[]
miss_identification_props=[]

model=VAE()
model.load_state_dict(torch.load("./current_model12v2"))
counter=0
counter2=0
for i in range(20000):
    loss=model.training_step(train)
    if counter==10000:
        test=torch.utils.data.DataLoader(d1,batch_size=len(d1), shuffle=False)
        a, b, c=model.test(test)
        correct_bodies.append(a)
        miss_identification_bodies.append(b)
        miss_identification_props.append(c)
        print(i, loss)
        print('percentage bodies correct',correct_bodies[-1][0]/sum(correct_bodies[-1][:2]))
        print('percentage props correct',correct_bodies[-1][2]/sum(correct_bodies[-1][2:4])) 
        model.scheduler.step()
        counter=0
    counter+=1
    if counter2==100:
        
        print(i, loss)
        counter2=0
    counter2+=1
    # model.scheduler.step()


torch.save(model.state_dict(), 'current_model12v3')