from DVAE import VAE
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import utils as util
import copy
import time
import torch



BS=10*1028
percent_train=0.8
# d1=util.create_dataset()
# torch.save(d1,'data12122023.pt')
d1=torch.load('data12122023.pt')#smd.run_sim(run_nums=2,out_data=2,num_repeats=1)
train=torch.utils.data.DataLoader(d1,batch_size=BS, shuffle=True)
test=torch.utils.data.DataLoader(d1,batch_size=len(d1), shuffle=False)
correct_bodies=[]
miss_identification_bodies=[]
miss_identification_props=[]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=VAE()
model.to(device)
# model.load_state_dict(torch.load('./model_12122023v4',map_location=torch.device('cpu')))
counter=0
counter2=0
loss_rec=[]
# t0=time.perf_counter()
for i in range(20000):
    loss=model.training_step(train,device)
    loss_rec.append(loss)
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
    print(i, loss)
    if counter2==1000:
        
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
torch.save(model.state_dict(), 'model_12122023_updated_weightv1')




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