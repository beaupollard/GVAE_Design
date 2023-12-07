import numpy as np
import math
import copy
from scipy.spatial.transform import Rotation as R
from gstools import SRF, Gaussian
import random

def build_steps(sim,num_steps=15,step_height=6.5/39.37,slope=30):
    b0=[]
    for i in range(num_steps):
        b0.append(sim.createPrimitiveShape(sim.primitiveshape_cuboid,[1.,2.,step_height]))
        l=step_height/math.tan(slope*math.pi/180)
        sim.setObjectPosition(b0[-1],b0[-1],[-l*i-1.0,0,step_height/2+step_height*i])



    final_pos=[(-l*(i+1)-0.5),step_height*i]
    Rw=sim.groupShapes(b0)
    
    
    # sim.setObjectInt32Param(Rw,sim.shapeintparam_static,0)
    sim.setObjectInt32Param(Rw,sim.shapeintparam_respondable,1)
    sim.cameraFitToView(0)
    return final_pos, step_height, slope, Rw

def build_slope(sim, slope=30):
    b0=[]
    for i in range(2):
        b0.append(sim.createPrimitiveShape(sim.primitiveshape_cuboid,[10,4.,0.05]))
        l=2/math.tan(slope*math.pi/180)
        sim.setObjectPosition(b0[-1],b0[-1],[-1,0,0])

    final_pos=[-5.3, 0]
    
    Rw=sim.groupShapes(b0)
    sim.setShapeColor(b0[-1],'',sim.colorcomponent_ambient_diffuse,[0.8,0.8,0.8])

    
    m=sim.getObjectMatrix(b0[-1],-1)
    axis = [m[1],m[5],m[9]]
    axisPos = sim.getObjectPosition(b0[-1], -1)

    m=sim.rotateAroundAxis(m,axis,axisPos,slope*math.pi/180)
    sim.setObjectMatrix(b0[-1],-1,m)

    sim.setObjectInt32Param(Rw,sim.shapeintparam_respondable,1)
    sim.cameraFitToView(0)
    return final_pos, slope, Rw

def flatten(l):
    return [item for sublist in l for item in sublist]

def build_gaussian_field(sim, num=20170519):
    x = y = range(256)

    model = Gaussian(dim=2, var=1, len_scale=10)
    srf = SRF(model, seed=num)

    field = srf.structured([x, y])
    field = field.tolist()

    field = flatten(field)
    smallest = min(field)
    field = list(map(lambda x: x / 10, field))

    b0 = []
    b0.append(sim.createHeightfieldShape(0, 3, 256, 256, 15, field))
    sim.setObjectPosition(b0[-1],b0[-1],[-5,0,0.1])
    # b0.append(sim.createHeightfieldShape(0, 3, 256, 256, 15, field))
    # sim.setObjectPosition(b0[-1],b0[-1],[-3,0,0])
    Rw=b0[-1]#sim.groupShapes(b0)

    # sim.setShapeColor(b0[-1],'',sim.colorcomponent_ambient_diffuse,[1,1,1])
    # sim.setShapeColor(b0[-2],'',sim.colorcomponent_ambient_diffuse,[1,1,1])


    final_pos = [-12., 0]

    sim.setObjectInt32Param(Rw,sim.shapeintparam_respondable,1)
    sim.cameraFitToView(0)
    return final_pos, 0, 0, Rw#b0[0]

def build_rough_slope(sim, slope=25, num=20170519):
    x = y = range(256)

    model = Gaussian(dim=2, var=1, len_scale=10)
    srf = SRF(model, seed=num)

    field = srf.structured([x, y])
    field = field.tolist()

    field = flatten(field)
    smallest = min(field)
    field = list(map(lambda x: x / 35, field))

    b0 = []
    b0.append(sim.createHeightfieldShape(0, 3, 256, 256, 5, field))
    sim.setObjectPosition(b0[-1],b0[-1],[-1,0,0])
    for i in range(1):
        b0.append(sim.createPrimitiveShape(sim.primitiveshape_cuboid,[10,6.,0.05]))
        l=0.46/math.tan(slope*math.pi/180)
        sim.setObjectPosition(b0[-1],b0[-1],[2.5,0,0])
    sim.setShapeColor(b0[-1],'',sim.colorcomponent_ambient_diffuse,[0.8,0.8,0.8])

    b1 = sim.createPrimitiveShape(sim.primitiveshape_cuboid,[1,1,1])
    sim.setObjectPosition(b1, b1, [-2,0,1])

    final_pos=[-5.3, 0]
    Rw=sim.groupShapes(b0)
    m=sim.getObjectMatrix(b1,-1)
    axis = [m[1],m[5],m[9]]
    axisPos = sim.getObjectPosition(b1, -1)
    
    m=sim.rotateAroundAxis(m,axis,axisPos,slope*math.pi/180)

    sim.setObjectMatrix(b0[-1],-1,m)
    
    sim.setObjectMatrix(b0[-2],-1,m)


    
    sim.removeObject(b1)
    sim.setObjectInt32Param(b0[-1],sim.shapeintparam_respondable,1)
    sim.cameraFitToView(0)
    return final_pos, 0.46, slope, b0