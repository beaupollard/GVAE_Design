from DVAE import VAE
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import utils as util
import copy
import time
import torch



BS=2*1028
percent_train=0.8
# d1=util.create_dataset()
# torch.save(d1,'data1.pt')
d1=torch.load('data1.pt')#smd.run_sim(run_nums=2,out_data=2,num_repeats=1)
train=torch.utils.data.DataLoader(d1,batch_size=BS, shuffle=True)

correct_bodies=[]
miss_identification_bodies=[]
miss_identification_props=[]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=VAE()

# model.load_state_dict(torch.load("./current_model2"))
counter=0
counter2=0
for i in range(2000):
    loss=model.training_step(train)
    if counter==10:
        # test=torch.utils.data.DataLoader(d1,batch_size=len(d1), shuffle=False)
        test=torch.utils.data.DataLoader(d1,batch_size=len(d1), shuffle=False)
        a, b, c=model.test(test)
        correct_bodies.append(a)
        miss_identification_bodies.append(b)
        miss_identification_props.append(c)
        print(i, loss)
        print('percentage bodies correct',correct_bodies[-1][0]/sum(correct_bodies[-1][:2]))
        print('percentage props correct',correct_bodies[-1][2]/sum(correct_bodies[-1][2:4])) 
        print('percentage joints correct',correct_bodies[-1][4]/sum(correct_bodies[-1][4:])) 
        counter=0
    counter+=1
    if counter2==100:
        
        model.scheduler.step()
        counter2=0
    counter2+=1

torch.save(model.state_dict(), 'current_model')




# for j in range(10):
#     model=VAE()
#     # model.load_state_dict(torch.load("./current_model33"))
#     counter=0
#     counter2=0
#     for i in range(500):
#         loss=model.training_step(train)
#         if counter==10:
#             # test=torch.utils.data.DataLoader(d1,batch_size=len(d1), shuffle=False)
#             test=torch.utils.data.DataLoader(d1,batch_size=len(d1), shuffle=False)
#             a, b, c=model.test(test)
#             correct_bodies.append(a)
#             miss_identification_bodies.append(b)
#             miss_identification_props.append(c)
#             # print(i, loss)
#             # print('percentage bodies correct',correct_bodies[-1][0]/sum(correct_bodies[-1][:2]))
#             # print('percentage props correct',correct_bodies[-1][2]/sum(correct_bodies[-1][2:4])) 
#             # print('percentage joints correct',correct_bodies[-1][4]/sum(correct_bodies[-1][4:])) 
#             counter=0
#         counter+=1
#         if counter2==100:
            
#             model.scheduler.step()
#             counter2=0
#         counter2+=1
    
#     print('counter: ',j)
#     print('percentage bodies correct',correct_bodies[-1][0]/sum(correct_bodies[-1][:2]))
#     print('percentage props correct',correct_bodies[-1][2]/sum(correct_bodies[-1][2:4])) 
#     print('percentage joints correct',correct_bodies[-1][4]/sum(correct_bodies[-1][4:])) 
#     torch.save(model.state_dict(), 'current_model'+str(j))
# print("hey")