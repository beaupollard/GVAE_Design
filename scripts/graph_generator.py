import numpy as np
import copy
import random

def set_length_location(min_x=-100,BLprev=0,bodyw=0,bodyh=0,zloc=0,yloc=0):
    err=False
    while err==False:
        body_length=get_gauss_rand(12.,4.,6.,14.)/39.37
        prop_radius=get_gauss_rand(8.,4.,2.,12.)/39.37
        prop_location=[get_gauss_rand(0,5.,-body_length,body_length)/39.37,bodyw+0.1,-bodyh/2+0.05]
        joint_location=[body_length/2+1/39.37,0,get_gauss_rand(0.,5,-5,5)/39.37]
        R0=BLprev/2+2/39.37+body_length/2
        overlap=R0+prop_location[0]-prop_radius
        if overlap>min_x:
            err=True
    min_x=prop_location[0]+prop_radius
    BLprev=body_length
    return body_length, prop_radius, prop_location, joint_location, BLprev, min_x

def get_gauss_rand(in_mean,in_std=0,l_lim=-1000000,u_lim=1000000):
    outp = l_lim-1
    while outp<l_lim or outp>u_lim:
        outp = np.random.normal(in_mean,in_std)
    return outp

class graph_gens():
    def __init__(self):
        self.nodes=[]
        self.node_attributes=[]
        self.edge_index=[]
        self.body_nodes={"name":"body","location":[],"length": 0,"width":20/39.37,"height":12/39.37,"clearance":0,"childern": [],"parents": [],"index":0}
        self.joint_nodes={"name":"joint","location": [0, 0, 0],"orientation":[0,0,0],"active":[],"childern": [],"parents": [],"index":1}
        self.prop_nodes={"name":"prop","location": [0, 0, 0],"radius":0,"childern": [],"parents": [],"type":'none'}
        self.num_propulsors=0

    def generate_concept(self):
        ## Determine number of bodies ##
        body_num=random.randint(2, 4)

        prev_tracked=False      # set that the previous mechanism wasn't a track
        current_node=0          # current graph node
        min_x=10                 # where there will be overlap in wheels
        prop_types=['none','wheel','planet wheel','track']

        for i in range(body_num):
            ## Initialize the node types ##
            body=self.body_nodes
            joint=self.joint_nodes
            propulsors=self.prop_nodes            

            ## Set the body node parent (previous joint) ##
            if i==0:
                body['parents']=[]
            else:
                body['parents']=current_node-1

            ## Determine if the next joint is active ##
            if random.randint(0, 1)==1:
                joint['active']=[get_gauss_rand(100,25,50,200),get_gauss_rand(75,50,25,200)]
            else:
                joint['active']=[]
            joint['parents']=current_node


            ## Determine what type of propulsion ##
            if prev_tracked==True:  # if the previous body started a track mechanism
                propulsors["type"]='track'
                prev_tracked=False
                self.num_propulsors+=1
            else:
                ## Determine what type of mechanism is next [0=none, 1=wheel, 2=planet wheel, 3=track] ##
                if i==body_num:
                    propulsors["type"]=prop_types[random.randint(0, 2)]
                elif i==0:
                    propulsors["type"]=prop_types[3]#random.randint(1, 3)]
                else:
                    propulsors["type"]=prop_types[random.randint(0, 3)]
                
                if propulsors["type"]=='track':
                    prev_tracked=True
            
            body["length"], propulsors["radius"], propulsors["location"], joint["location"], BLprev, min_x = set_length_location(min_x=-100,BLprev=0,bodyw=body["width"],bodyh=body["height"],zloc=0,yloc=0)

            if propulsors["type"]==0:
                body["childern"]=[current_node+1]
                self.nodes.append(copy.deepcopy(body))
                self.nodes.append(copy.deepcopy(joint))
                current_node+=2
            else:
                body["childern"]=[current_node+1,current_node+2]
                propulsors["parents"]=current_node
                self.nodes.append(copy.deepcopy(body))
                self.nodes.append(copy.deepcopy(propulsors))      
                self.nodes.append(copy.deepcopy(joint))  
                self.num_propulsors+=1              
                current_node+=3
        return self.nodes
            
con=graph_gens()
con.generate_concept()