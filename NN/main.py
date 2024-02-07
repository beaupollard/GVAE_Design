from DVAE import VAE
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import utils as util
import copy
import time
import torch
import sys
import wandb

random.seed(10)
BS=10*1028
percent_train=0.8
# d1=util.create_dataset()
# torch.save(d1,'data_from EA.pt')
d1=torch.load('data_updated_CNN.pt')#smd.run_sim(run_nums=2,out_data=2,num_repeats=1)
# d1=torch.load('data12122023.pt')#smd.run_sim(run_nums=2,out_data=2,num_repeats=1)
del_index=[]
for i in range(len(d1)):
    
    if d1[i][1].detach().numpy()@np.array([-0.15,-0.15,0.7])<0.:
        del_index.append(i)
del_index.reverse()
for i in del_index:
    del d1[i]

random.shuffle(d1)
train_size=int(len(d1)*percent_train)
train=torch.utils.data.DataLoader(d1[:train_size],batch_size=BS, shuffle=True)
test=torch.utils.data.DataLoader(d1[train_size:],batch_size=len(d1[train_size:]), shuffle=False)
correct_bodies=[]
miss_identification_bodies=[]
miss_identification_props=[]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


print ('argument list', sys.argv)
seed_in = "0"#sys.argv[1]



model=VAE(seed=int(seed_in),lr=2e-3,latent_dim=20)
model.to(device)
# model.load_state_dict(torch.load('model_01192024_pp_reg'+seed_in))
wandb.init(
    project='DVAE Updated ze LD16 encoder conditioned on y',
    config={
        "seed": seed_in,
        "latent_dim": model.latent_dim, 
        "learning_rate": model.lr
    }
)
init_kl_weight=copy.deepcopy(model.kl_weight)
init_pp_weight=copy.deepcopy(model.perf_weight)
# model.load_state_dict(torch.load('./model_12122023v4',map_location=torch.device('cpu')))
counter=0
counter2=0
loss_rec=[]
# t0=time.perf_counter()
for i in range(40000):
    loss=model.training_step(train,device)
    loss_rec.append(loss)
    loss_val=model.test_val(test,device)
    if counter==500:
        # test=torch.utils.data.DataLoader(d1,batch_size=len(d1), shuffle=False)
        model.to("cpu")
        # correct_bodies, miss_identification_bodies, miss_identification_propsc=model.test(test,"cpu")       
        a, b, c=model.test(test,"cpu")
        correct_bodies.append(a)
        miss_identification_bodies.append(b)
        miss_identification_props.append(c)
        
        print('percentage bodies correct',correct_bodies[-1][0]/sum(correct_bodies[-1][:2]))
        print('percentage props correct',correct_bodies[-1][2]/sum(correct_bodies[-1][2:4])) 
        print('percentage joints correct',correct_bodies[-1][4]/sum(correct_bodies[-1][4:])) 
        counter=0
        model.to(device)
    # model.test_pp(test,device)
    wandb.log({"epoch": i, "train reals loss": loss[0], "train ints loss": loss[1], "train KL div": loss[2]/model.kl_weight,"train Perf loss": loss[3]/model.perf_weight})
    wandb.log({"val reals loss": loss_val[0], "val ints loss": loss_val[1], "val KL div": loss_val[2]/model.kl_weight,"val Perf loss": loss_val[3]/model.perf_weight})
    # print(i, loss)
    # model.kl_weight=init_kl_weight*(1+10*(1-math.exp(-i/10000)))
    # model.perf_weight=init_pp_weight*(1+10*(1-math.exp(-i/10000)))

    if counter2==2000:
        
        model.scheduler.step()
        counter2=0
    counter+=1
    counter2+=1
    # timer=time.perf_counter()-t0
    # t0=time.perf_counter()
    # print(timer)    
    # print(i, loss, timer)
    # timer=time.perf_counter()-t0
    # t0=time.perf_counter()
    # print(timer) current_model2 has a latent space size of 16
torch.save(model.state_dict(), 'model_02072024_LD20_condition_'+seed_in)




# for j in range(10):
#     model=VAE()
#     model.to(device)
#     # model.load_state_dict(torch.load("./current_model33"))
#     counter=0
#     counter2=0
#     for i in range(2000):
#         loss=model.training_step(train,device)
#         counter+=1
#         if counter2==100:
            
#             model.scheduler.step()
#             counter2=0
#         counter2+=1
#     model.to("cpu")
#     correct_bodies, miss_identification_bodies, miss_identification_propsc=model.test(test,"cpu")   
#     print('counter: ',j)
#     print('percentage bodies correct',correct_bodies[0]/sum(correct_bodies[:2]))
#     print('percentage props correct',correct_bodies[2]/sum(correct_bodies[2:4])) 
#     print('percentage joints correct',correct_bodies[4]/sum(correct_bodies[4:])) 
#     torch.save(model.state_dict(), 'current_model'+str(j))
# print("hey")