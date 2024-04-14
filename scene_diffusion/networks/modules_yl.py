
import math
from random import random
from functools import partial
from collections import namedtuple
from tkinter.messagebox import NO
from tkinter.tix import Tree

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm
from .denoise_net import *

prt = False
visual = False

"""
class relaYL():
    def __init__(self, config):
        #all kinds of dimensions here
        self.batchsz = config.get("batchsz", 128)

        #................basic length of the data structure...................
            #.........absoluteTensor.....................
        self.objectness_dim = config.get("objectness_dim", 0) #exist or not, if objectness_dim=0, 
                                            #then we use the last digit of class_dim (or "end_label") as the objectness
        self.class_dim = config.get("class_dim", 21)
        self.translation_dim = config.get("translation_dim", 3)
        self.size_dim = config.get("size_dim", 3)
        self.angle_dim = config.get("angle_dim", 2)
        self.bbox_dim = self.translation_dim + self.size_dim + self.angle_dim
        self.objfeat_dim = config.get("objfeat_dim", 0)
        self.point_dim = config.get("point_dim", 62)
        self.maxObj = config.get("sample_num_points", 12)
        
        if self.point_dim != self.bbox_dim+self.class_dim+self.objectness_dim+self.objfeat_dim:
            raise NotImplementedError

            #.........wallTensor.....................
        self.wall_translation_dim = config.get("wall_translation_dim", 2)
        self.wall_norm_dim = config.get("wall_norm_dim",2)
        self.wall_dim = self.wall_translation_dim + self.wall_norm_dim
        self.maxWall = config.get("maxWall", 16)

            #.........windoorTensor.....................
        self.windoor_translation_dim = config.get("windoor_translation_dim", 3)
        self.windoor_scale_dim = config.get("windoor_scale_dim", 3)
        self.windoor_norm_dim = config.get("windoor_norm_dim", 2)
        self.windoor_dim = self.windoor_translation_dim + self.windoor_scale_dim + self.windoor_norm_dim
        self.maxWindoor = config.get("maxWindoor",8)

        #^^^^^^^^^^^^^^^^^^^^^basic length of the data structure^^^^^^^^^^^^^^^^^^^

        #.....................calculating process..................................
            #.............distance()
        self.relativeTranslation = config.get("relativeTranslation", True)
                #the form of relative translation
        self.relativeOrien_Trans = config.get("relativeOrien_Trans", True)
        self.relativeScl_Ori_Trn = config.get("relativeScl_Ori_Trn", True)

        self.relativeOrientation = config.get("relativeOrientation", True)
                #the form of relative angle: of course, their difference (or their rotating matrix, it's up to "self.angle_dim")

        self.relativeScale       = config.get("relativeScale", True)
                #the form of relative scale
        self.relativeScaleMethod = config.get("relativeScaleMethod", "minus")

                #the form of relative class labels ?????
        self.relativeClassLabels = config.get("relativeClassLabels", True)
        
                #the form of relative object feature ???  this could even become an interesting topic
        self.relativeObjFeatures = config.get("relativeObjFeatures", False)
        
                #the way to add these relative values together
        self.weightForm = config.get("weightForm", "exp-dis")
        self.alpha = config.get("diminish_coefficient", 1.0)
        self.nearestNum = config.get("nearestNum", 1)

            #.............wall()
        self.relative_wall_translation = config.get("relative_wall_translation", True)
        self.relative_wall_orientation = config.get("relative_wall_orientation", True)
        self.relative_wall_scale = config.get("relative_wall_scale", False)
        
            #.............windoor()
        self.relative_windoor_translation = config.get("relative_windoor_translation", True)
        self.relative_windoor_orientation = config.get("relative_windoor_orientation", True)
        self.relative_windoor_scale = config.get("relative_windoor_scale", False)

        if self.angle_dim != 2 or self.wall_norm_dim != 2 or self.windoor_norm_dim != 2:
            raise NotImplementedError
        pass
    
    @staticmethod
    def norm_Minus(normA, normB):
        cs = (normA * normB).sum(axis=-1)
        na = normA.reshape((-1,2))
        nA = torch.cat([na[:,1:], -na[:,:1]], axis=-1).reshape(normA.shape)
        sn = (normB * nA).sum(axis=-1)
        return torch.cat([cs,sn],axis=-1).reshape((normA.shape[0], max(normA.shape[1],normB.shape[1]), max(normA.shape[2],normB.shape[2]), 2))

    @staticmethod
    def simple_Norm_Minus(normA, normB):
        cs = (normA * normB).sum(axis=-1)
        na = normA.reshape((-1,2))
        nA = torch.cat([na[:,1:], -na[:,:1]], axis=-1).reshape(normA.shape)
        sn = (normB * nA).sum(axis=-1)
        return torch.cat([cs,sn],axis=-1).reshape(normA.shape)

    @staticmethod
    def theta2norm(theta):
        return torch.cat([torch.cos(theta), torch.sin(theta)], axis=-1)
    
    @staticmethod
    def theta_Minus(thetaA, thetaB):
        return thetaA - thetaB #cast from -pi to pi

    def windoor(self, absoluteTensor, windoorTensor):
        # batchsz = 128 : maxObj = 12 : bbox_dim = 8         absolutionTensor.dim = -1:
        # (trans_x, ..y, ..z, scale_x, ..y, ..z, norm_Z(cosA), norm_X(sinA) ) #Z before X in norm

        # batchsz = 128 : maxW = 8 : point_dim = 8         windoorTensor.dim = -1:
        # (midTranslation_dim=3)+(scale_dim=3)+(innerNorm_dim=2) 

        xx = absoluteTensor.reshape((self.batchsz,self.maxObj,1,-1))
        if prt:
            print(xx.shape)

        yy = windoorTensor.reshape((self.batchsz,1,-1,self.windoor_dim))
        if prt:
            print(yy.shape)

        if not self.relative_wall_translation:
            raise NotImplementedError
        #self.relative_wall_translation == True: #wall translation minus object translation

        tZ = yy[:,:,:,self.windoor_translation_dim-1:self.windoor_translation_dim] - xx[:,:,:,self.translation_dim-1:self.translation_dim]
        tX = yy[:,:,:,0:1] - xx[:,:,:,0:1]
        swaptCo_ord = torch.cat([tZ, tX], axis=-1)

        dis = (swaptCo_ord**2).sum(axis=-1)
        dis = dis.reshape((self.batchsz, xx.shape[1], yy.shape[2]))
        if prt:
            print("dis")
            print(dis.shape)
            print(dis.tolist())

        if not self.relative_wall_orientation:
            raise NotImplementedError
        #self.relative_wall_orientation == True: #rotate with the object angle

        #theta in source object's co-ordinates        
        if self.angle_dim == 2:
            swaprtCo_ord = self.norm_Minus(swaptCo_ord, xx[:,:,:,self.bbox_dim-self.angle_dim: self.bbox_dim])
        else: #self.angle_dim == 1
            raise NotImplementedError #still buggy!!!!
            tCo_ord = torch.arctan2(relX,relZ).reshape((self.batchsz,self.maxObj,-1,1)) - xx[:,:,:,self.bbox_dim-self.angle_dim:self.bbox_dim]

        swaprtCo_ord = swaprtCo_ord.reshape((self.batchsz, xx.shape[1], yy.shape[2], self.angle_dim))
        if prt:
            print("swaprtCo_ord")
            print(swaprtCo_ord.shape)
            print(swaprtCo_ord.tolist())

        relCo_ord = torch.cat([swaprtCo_ord[:,:,:,-1:], swaprtCo_ord[:,:,:,1:]], axis = -1)

        if self.angle_dim == 2: 
            if self.windoor_norm_dim == 2:
                windoor_norm = yy[:,:,:,self.windoor_dim-self.windoor_norm_dim:self.windoor_dim]
            else:
                raise NotImplementedError
                windoor_norm = self.theta2norm(yy[:,:,:,self.windoor_dim-self.windoor_norm_dim:self.windoor_dim])
            
            rWNV = self.norm_Minus(windoor_norm, xx[:,:,:,self.bbox_dim-self.angle_dim:self.bbox_dim])
        else: #self.angle_dim = 1
            raise NotImplementedError
            if self.windoor_norm_dim == 2:
                raise NotImplementedError
                #arctan2
            else:
                windoor_angle = yy[:,:,:,self.windoor_dim-self.windoor_norm_dim:self.windoor_dim]
            
            rWNV = (windoor_angle - xx[:,:,:,self.bbox_dim-self.angle_dim:self.bbox_dim])
        
        rWNV = rWNV.reshape((self.batchsz, xx.shape[1], yy.shape[2], self.angle_dim))
        if prt:
            print("rWNV")
            print(rWNV.shape)
            #print(rWNV.tolist())

        relTheta = rWNV

        if self.relative_wall_scale:
            #self.relative_wall_scale == True: #scale the relative translation and the relative norm-vector
            swapScale = torch.cat([xx[:,:,:,self.translation_dim+self.size_dim-1:self.translation_dim+self.size_dim], xx[:,:,:,self.translation_dim:self.translation_dim+1]], axis=-1) #[scale_z, scale_x]
            swapsrtCo_ord = swaprtCo_ord / swapScale 
            srWNV = rWNV * swapScale
            relTheta = F.normalize(srWNV, dim=-1)
            relCo_ord = torch.cat([swapsrtCo_ord[:,:,:,-1:], swapsrtCo_ord[:,:,:,1:]], axis = -1)
            dis = (swapsrtCo_ord**2).sum(axis=-1)
            dis = dis.reshape((self.batchsz, xx.shape[1], yy.shape[2]))

        #then we use the mixing step--------------------------dis, relCo_ord, relTheta-------------------------

        mins, argmins = dis.min(axis=-1)
        if prt or 1:
            print("mins")
            print(mins.shape)
            #print(mins.tolist())
            print("argmins")
            print(argmins.shape)
            print(argmins.tolist())
            #print(torch.arange(batchsz).reshape((-1,1)).repeat(1, argmins.shape[1]).shape)
            #print(torch.arange(argmins.shape[1]).reshape((1,-1)).repeat(batchsz, 1).shape)

        id0 = torch.arange(self.batchsz  ).reshape((-1,1,1)).repeat(1,self.maxObj,self.angle_dim)
        id1 = torch.arange(self.maxObj   ).reshape((1,-1,1)).repeat(self.batchsz,1,self.angle_dim)
        id2 = argmins.reshape((self.batchsz,self.maxObj,1)).repeat(1,1,self.angle_dim)
        id3 = torch.arange(self.angle_dim).reshape((1,1,-1)).repeat(self.batchsz,self.maxObj,1)
        #thetamin.shape = (batchsz=128) : (maxObj=12) : (angle_dim=1/2)

        thetamin = relTheta[id0, id1, id2, id3]
        if prt or 1:
            print("thetamin")
            print(thetamin.shape)
            #print(thetamin.tolist())

        comin = relCo_ord[id0, id1, id2, id3]
        if prt or 1:
            print("comin")
            print(comin.shape)
            print(comin.tolist())
        
        resultTensor = torch.cat([comin.reshape((self.batchsz,-1,2)), thetamin.reshape((self.batchsz,-1,self.angle_dim))],axis = -1)
        
        # batchsz = 128 : maxObj = 12 : point_dim = 3/4       resultTensor.dim = -1:
        # co_ord(2) + theta(1)/dire(2)    #this result is not enough, there still other things to be done
        return resultTensor   #resultTensor = absoluteTensor

    def wall(self, absoluteTensor, wallTensor):
        # batchsz = 128 : maxObj = 12 : bbox_dim = 8         absolutionTensor.dim = -1:
        # (trans_x, trans_y, trans_z, scale_x, scale_y, scale_z, norm_Z(cosA), norm_X(sinA) )

        # batchsz = 128 : maxW = 16 : point_dim = 4         wallTensor.dim = -1:
        # point_x, point_z, innerNorm_x, innerNorm_z 

        wal = torch.cat([wallTensor[:,-1:], wallTensor[:,:-1]], axis = -2)
        if prt:
            print(wal.shape)
            print(wal.tolist())
            print(wal.shape)
        wallStart = wal[:,:,:self.wall_translation_dim]
        #lenSquare = ((wallTensor[:,:,:self.wall_translation_dim] - wallCenter[:,:,:self.wall_translation_dim])**2).sum(axis=-1).reshape((self.batchsz,self.maxWall,1))
        if prt:
            print("lenSquare")
            #print(lenSquare.shape)
            #print(lenSquare.tolist())
        wallNorm = wal[:,:,self.wall_translation_dim:self.wall_translation_dim+self.wall_norm_dim]
        if prt:
            print("wallNorm")
            print(wallNorm.shape)
            print(wallNorm.tolist())
        #wallDire = torch.cat([ wal[:,:,self.wall_translation_dim+self.wall_norm_dim-1:self.wall_translation_dim+self.wall_norm_dim], -wal[:,:,self.wall_translation_dim:self.wall_translation_dim+1] ], axis=-1)
        #if prt:
            #print("wallDire")
            #print(wallDire.shape)
            #print(wallDire.tolist())

        xx = absoluteTensor.reshape((self.batchsz,self.maxObj,1,-1))
        if prt:
            print("xx")
            print(xx.shape)

        WS = wallStart.reshape((self.batchsz,1,self.maxWall,self.wall_translation_dim))
        if prt:
            print("WC")
            print(WC.shape)

        #the problem right now is to re-organize this stuff

        #we have what?
            #Under World Co-ordinates
                #Object-Translation     Object-size     Object-Normalization
                #Wall-Center, Wall-Normalization, Wall-Length

            #finally, we can calculate the distance, shift, bound, orientation under "this co-ordinates"

        #during the process, we maintains: relative wallCenter, relative wallNormVecter, relative wallTangVector
        #at the end, we calculate: distance and orientation, shift&bound with the wC, wNV, wTV

        if not self.relative_wall_translation:
            raise NotImplementedError
        #self.relative_wall_translation == True: #wall translation minus object translation

        tWS = torch.cat([ xx[:,:,:,0:1] - WS[:,:,:,0:1], xx[:,:,:,self.translation_dim-1:self.translation_dim] - WS[:,:,:,self.wall_translation_dim-1:self.wall_translation_dim] ], axis=-1)
        if prt:
            print("tWC")
            print(tWC.shape)
            print(tWC.tolist())
        swaptWS = torch.cat([tWS[:,:,:,1:2], tWS[:,:,:,0:1]], axis=-1)
        
        dis = (swaptWS * wallNorm.reshape((self.batchsz,1,-1,self.wall_norm_dim))).sum(axis=-1)
        if prt:
            print("dis")
            print(dis.shape)
            print(dis.tolist())

        #swaptWS[0] * wallNorm[1] - swaptWS[1] * wallNorm[0]
        aaa = self.norm_Minus(swaptWS, wallNorm.reshape((self.batchsz,1,-1,self.wall_norm_dim)))[:,:,:,1] 
        
        #lengthSquare = (swaptWC * wallDire.reshape((self.batchsz,1,-1,self.wall_norm_dim))).sum(axis=-1)**2
        #if prt:
            #print("lengthSquare")
            #print(lengthSquare.shape)
            #print(lengthSquare.tolist())

        #一方面，无论是否进行旋转，dis的值都不会变，（如果需要进行仿射，则dis还需要重置）
        #另外，无论是否需要进行旋转和仿射，实际上shift和bound之间的关系都应当是确定的，因此我们在这里先把shift和bound定下来了

        if not self.relative_wall_orientation:
            raise NotImplementedError
        #self.relative_wall_orientation == True: #rotate with the object angle
        
        #wNV rotate with -angle
        if self.angle_dim == 1:
            raise NotImplementedError
            rWNV = torch.arctan2( wallNorm[:,:,1:2] , wallNorm[:,:,0:1] ).reshape((batchsz,1,-1,self.angle_dim)) - xx[:,:,:,self.bbox_dim-self.angle_dim:self.bbox_dim].reshape((batchsz,-1,1,self.angle_dim))
        else:
            rWNV = self.norm_Minus(wallNorm.reshape(self.batchsz,1,-1,self.wall_norm_dim), xx[:,:,:,self.bbox_dim-self.angle_dim:self.bbox_dim])
        if prt:#or True:
            print("rWNV")
            print(rWNV.shape)
            print(rWNV.tolist())

        relTheta = rWNV

        if self.relative_wall_scale:
            #self.relative_wall_scale == True: #scale the relative translation and the relative norm-vector
            #wC rotate with -angle (we should form a matrix for it)
            if self.angle_dim == 1:
                raise NotImplementedError
            else:
                swaprtWS = self.norm_Minus(swaptWS,xx[:,:,:,self.bbox_dim-self.angle_dim:self.bbox_dim])
            
            swapScale = torch.cat([xx[:,:,:,self.translation_dim+self.size_dim-1:self.translation_dim+self.size_dim], xx[:,:,:,self.translation_dim:self.translation_dim+1]], axis=-1) #[scale_z, scale_x]
            swapsrtWS = swaprtWS / swapScale 
            srWNV = rWNV * swapScale
            relTheta = F.normalize(srWNV, dim=-1)
            dis = (swapsrtWS * relTheta).sum(axis=-1)
            aaa = self.norm_Minus(swapsrtWS, srWNV.reshape((self.batchsz,1,-1,self.wall_norm_dim)))[:,:,:,1] 
        
        #then we use the mixing step----------------(dis: (batchsz, maxObj, maxW, 1), relTheta: (batchsz, maxObj, maxW, 2?))------------------------------

        aaabbb = aaa * torch.cat([aaa[:,:,1:],aaa[:,:,:1]], axis=-1)
        cond = aaabbb < torch.zeros_like(aaabbb)
        #cond = lengthSquare < lenSquare.reshape((self.batchsz,1,-1))
        if prt:
            print(cond.shape)
            print(cond.tolist())
        res = torch.ones_like(dis) * 10000
        res[cond] = dis[cond]
        if prt:
            print(res.shape)

        mins, argmins = res.min(axis=-1)
        if prt:
            print(mins)
            print(mins.shape)
            print(mins.tolist())
            print(argmins.shape)
            print(argmins.tolist())
            print(torch.arange(batchsz).reshape((-1,1)).repeat(1, argmins.shape[1]).shape)
            print(torch.arange(argmins.shape[1]).reshape((1,-1)).repeat(batchsz, 1).shape)

        #argmins.shape = (batchsz=128) : (maxObj=12) : (result=1)
        #index tensor is: for a mother tensor with N dimensions, we need N index tensor with same shape
        #mother[ indexTensor_0, indexTensor_1, ...... indexTensor_N-1]
        #while the shape of these indexTensor turns out to be the shape of resultTensor
        #the value across each indexTensor turns out to be the index of each element
        #now lets go for our own case

        #relTheta.shape = (batchsz=128) : (maxObj=12) : (maxWall=16) : (angle_dim=1/2)
        #                00..11...127127  0123...0123.. argmins repeat 01010101010101
        #                  (128*12*1/2) : (128*12*1/2) : (128*12*1/2) : (128*12*1/2)

        id0 = torch.arange(self.batchsz  ).reshape((-1,1,1)).repeat(1,self.maxObj,self.angle_dim)
        id1 = torch.arange(self.maxObj   ).reshape((1,-1,1)).repeat(self.batchsz,1,self.angle_dim)
        id2 = argmins.reshape((self.batchsz,self.maxObj,1)).repeat(1,1,self.angle_dim)
        id3 = torch.arange(self.angle_dim).reshape((1,1,-1)).repeat(self.batchsz,self.maxObj,1)
        #thetamin.shape = (batchsz=128) : (maxObj=12) : (angle_dim=1/2)

        thetamin = relTheta[id0, id1, id2, id3]
        if prt:
            print(thetamin.shape)
            print(thetamin.tolist())

        resultTensor = torch.cat([mins.reshape((self.batchsz,-1,1)),thetamin.reshape((self.batchsz,-1,self.angle_dim))],axis = -1) #,argmins.reshape((batchsz,-1,1))
        #print(resultTensor.shape)

        # batchsz = 128 : maxObj = 12 : point_dim = 2/3         resultTensor.dim = -1:
        # min_Distance2 + Orientation(1) / Norm(2)                       #resultTensor = absoluteTensor
        return resultTensor

    def distance(self, absoluteTensor):
        # batchsz = 128 : maxObj = 12 : point_dim = 6?         absolutionTensor.dim = -1:
        # (translation_dim=3)+(size_dim=3)+(angle_dim=2) + (class_dim = 2?) + (objfeat_dim = 32)
        # (             bbox_dim   =   8    cosA, sinA )
        # (                          point_dim     =     6?                                    )
        xx = absoluteTensor.reshape((self.batchsz,-1,1,self.point_dim))
        if prt:
            print("xx")
            print(xx.shape)
        #xx.shape    (batchsz=128) : (src_dim=maxObj=12) : (dst_dim=1) : (translation_dim + angle_dim=4)

        yy = absoluteTensor.reshape((self.batchsz,1,-1,self.point_dim))
        if prt:
            print("yy")
            print(yy.shape)
        #yy.shape    (batchsz=128) : (src_dim=1) : (dst_dim=maxObj=12)  : (translation_dim + angle_dim=4)

        rel = (yy[:,:,:,:self.translation_dim] - xx[:,:,:,:self.translation_dim]).reshape(self.batchsz,self.maxObj,-1,1,self.translation_dim)
        if prt:
            print("rel")
            print(rel.shape)
            print(rel.tolist())
        #rel.shape    (batchsz=128) : (src_dim=maxObj=12) : (dst_dim=maxObj=12) : (spatial_dim_hori=1) : (spatial_dim_vert=3)  #vertical vector in spatial space

        if self.angle_dim == 1:
            relTheta = xx[:,:,:,self.bbox_dim-self.angle_dim:self.bbox_dim] - yy[:,:,:,self.bbox_dim-self.angle_dim:self.bbox_dim]
            #relTheta.shape (batchsz=128) : (src_dim=maxObj=12) : (dst_dim=maxObj=12) : (angle_dim=1)

            cs = torch.cos(xx[:,:,:,self.bbox_dim-self.angle_dim:self.bbox_dim]).reshape((self.batchsz,-1,1,1,1))
            sn = torch.sin(xx[:,:,:,self.bbox_dim-self.angle_dim:self.bbox_dim]).reshape((self.batchsz,-1,1,1,1))

            #cs.shape    (batchsz=128) : (src_dim=maxObj=12) : (dst_dim=1) : (spatial_dim_hori=1) : (spatial_dim_vert=1)  #scaler in spatial space
        else:
            relTheta = self.norm_Minus(xx[:,:,:,self.bbox_dim-self.angle_dim:self.bbox_dim], yy[:,:,:,self.bbox_dim-self.angle_dim:self.bbox_dim])

            cs = xx[:,:,:,self.bbox_dim-self.angle_dim:self.bbox_dim-self.angle_dim+1].reshape((self.batchsz,-1,1,1,1))
            sn = xx[:,:,:,self.bbox_dim-1:self.bbox_dim].reshape((self.batchsz,-1,1,1,1))

        rotateX = torch.cat([cs, torch.zeros(cs.shape, device=cs.device), sn], axis = -1)
        #rotateX.shape    (batchsz=128) : (src_dim=maxObj=12) : (dst_dim=1) : (spatial_dim_hori=1) : (spatial_dim_vert=3)  #vertical vector in spatial space
        rotateY = torch.cat([torch.zeros(cs.shape, device=cs.device), torch.ones(cs.shape, device=cs.device), torch.zeros(cs.shape, device=cs.device)], axis = -1)
        rotateZ = torch.cat([-sn, torch.zeros(cs.shape, device=cs.device), cs], axis = -1)

        rotate = torch.cat([rotateX, rotateY, rotateZ], axis = -2)
        #rotate.shape    (batchsz=128) : (src_dim=maxObj=12) : (dst_dim=1) : (spatial_dim_hori=3) : (spatial_dim_vert=3)  #3*3 matrix in spatial space
        newRel = (rotate * rel).sum(axis=-1)
        #newRel.shape    (batchsz=128) : (src_dim=maxObj=12) : (dst_dim=maxObj=12) : (spatial_dim_hori=3) : (spatial_dim_vert=1)  #horizontal vector in spatial space
        newRel = newRel.reshape((self.batchsz,-1,self.maxObj,3))
        #newRel.shape    (batchsz=128) : (src_dim=maxObj=12) : (dst_dim=maxObj=12) : (translation_dim=3) #flatten spatial space

        if prt:
            print("relTheta")
            print(relTheta.shape)
            print(relTheta.tolist())

        dis2 = (rel ** 2).sum(axis=-1).reshape((self.batchsz,self.maxObj,-1))
        if prt:
            print("dis2")
            print(dis2.shape)
            print(dis2.tolist())
        #dis2.shape    (batchsz=128) : (src_dim=maxObj=12) : (dst_dim=maxObj=12)

        expdis2 = torch.exp(-dis2)
        expdis2_0 = torch.zeros_like(expdis2)
        #expdis2.shape    (batchsz=128) : (src_dim=maxObj=12) : (dst_dim=maxObj=12)

        #ban掉那些class_dim末位是>0的那些index， 让他们的expdis2 = 0
        #absoluteTensor.shape = batchsz = 128 : maxObj = 12 : fullLength = 62
        
        end_label = absoluteTensor[:,:,self.bbox_dim+self.class_dim-1:self.bbox_dim+self.class_dim] #30???????

        #end_label.shape : (batchsz = 128) : (ambiguous_dim = maxObj = 12) : (label_dim = 1)
        cond = (end_label > torch.zeros_like(end_label)).reshape((self.batchsz, 1, self.maxObj))
        #cond.shape    (batchsz=128) : (src_dim=1) : (dst_dim=maxObj=12) 
        cond = cond.repeat((1, self.maxObj, 1))
        #cond.shape    (batchsz=128) : (src_dim=maxObj=12)  : (dst_dim=maxObj=12)
        expdis2[cond] = expdis2_0[cond]


        sumexpdis2 = expdis2.sum(axis=-1).reshape((self.batchsz,-1,1))
        #sumexpdis2.shape (batchsz=128) : (src_dim=maxObj=12)  : (dst_dim=1) # for each source, the destination of it are summed up 

        weights = (expdis2 / sumexpdis2).reshape((self.batchsz,self.maxObj,-1,1)) #(batchsz, 12, 12)
        #weights.shape    (batchsz=128) : (src_dim=maxObj=12)  : (dst_dim=maxObj=12) : (feat_dim=1) # to be broadcasted 

        #how to use this fucking weights
        #weights[i][j]  the j-th object weight to i-th  object

        #weights[0][0]      weights[0][1]       weights[0][2]=0     weights[0][3] 
        #weights[1][0]      weights[1][1]       weights[1][2]=0     weights[1][3]
        #weights[2][0]!=0   weights[2][1]!=0    weights[2][2]=0     weights[2][3]!=0
        #weights[3][0]      weights[3][1]       weights[3][2]=0     weights[3][3]

        #the problem is how to get the relative value of class_label and objfeats, these things can be controled by the config

        bb_sizes = absoluteTensor[:,:,self.translation_dim:self.translation_dim + self.size_dim]
        cl_class_labels = absoluteTensor[:,:,self.bbox_dim:self.bbox_dim+self.class_dim] #shape (batchsz=128) : (maxObj=12) : (feat_dim = 22)
        of_objfeats_32 = absoluteTensor[:,:,self.bbox_dim+self.class_dim:]

        xx_sizes = bb_sizes.reshape((self.batchsz,-1, 1, 3))
        yy_sizes = bb_sizes.reshape((self.batchsz, 1,-1, 3))

        if self.relativeScale:
            if self.relativeScaleMethod == "minus":
                relScale = (yy_sizes - xx_sizes).reshape((self.batchsz,self.maxObj,-1,self.size_dim))
            if self.relativeScaleMethod == "divide":
                relScale = (yy_sizes / xx_sizes).reshape((self.batchsz,self.maxObj,-1,self.size_dim))
        #relScale.shape    (batchsz=128) : (src_dim=maxObj=12)  : (dst_dim=maxObj=12) : (feat_dim = 3)
        else:
            relScale = xx_sizes.repeat((1,self.maxObj,1,1))

        if self.relativeScl_Ori_Trn:
            newRel = (newRel / xx_sizes).reshape((self.batchsz,-1,self.maxObj,3))

        relCL = cl_class_labels.reshape((self.batchsz,1,self.maxObj,-1)).repeat((1,self.maxObj,1,1)) #?????/
        relOF = of_objfeats_32.reshape((self.batchsz,1,self.maxObj,-1)).repeat((1,self.maxObj,1,1)) #?????/

        fullRel = torch.cat([newRel, relScale, relTheta, relCL, relOF], axis = -1)
        if prt:
            print("fullRel")
            print(fullRel.shape)
        #fullRel.shape    (batchsz=128) : (src_dim=maxObj=12)  : (dst_dim=maxObj=12) : (feat_dim = ???)

        resultTensor = (fullRel * weights).sum(axis=-2).reshape((self.batchsz, self.maxObj, -1)) #print(resultTensor.shape)
        return resultTensor  #resultTensor = absoluteTensor
    
    def rela(self, absoluteTensor, wallTensor, windoorTensor):
        self.batchsz = absoluteTensor.shape[0]
        print("absoluteTensor.shape")
        print(absoluteTensor.shape)
        print("wallTensor.shape")
        print(wallTensor.shape)

        relativeWall = self.wall(absoluteTensor[:,:,:self.bbox_dim], wallTensor)
        print("relativeWall.shape")
        print(relativeWall.shape)
        print("windoorTensor.shape")
        print(windoorTensor.shape)
        relativeWindoor = self.windoor(absoluteTensor[:,:,:self.bbox_dim], windoorTensor)
        print("relativeWindoor.shape")
        print(relativeWindoor.shape)
        relativeObject = self.distance(absoluteTensor)
        print("relativeObject.shape")
        print(relativeObject.shape)

        relativeTensor = torch.cat([relativeObject, relativeWall, relativeWindoor], axis=-1)

        print("relativeTensor.shape")
        print(relativeTensor.shape)
    
        end_label = absoluteTensor[:,:,self.bbox_dim+self.class_dim-1:self.bbox_dim+self.class_dim]
        #end_label.shape : (batchsz = 128) : (ambiguous_dim = maxObj = 12) : (label_dim = 1)
        cond = (end_label > torch.zeros_like(end_label))
        #cond.shape    (batchsz=128) : (src_dim=1) : (dst_dim=maxObj=12) 
        cond = cond.repeat((1, 1, relativeTensor.shape[-1]))
        #cond.shape    (batchsz=128) : (src_dim=maxObj=12)  : (dst_dim=maxObj=12)
        print("cond.shape")
        print(cond.shape)
        relativeZero = torch.zeros_like(relativeTensor)
        relativeTensor[cond] = relativeZero[cond]
        #别忘了ban掉那些class_dim末位是>0的那些index， 让他们的行掉为0

        print(relativeTensor.dtype)
        return relativeTensor

    def recon(self, absoluteTensor, denoise_out):
        #denoise_out is the difference in the objects' own co-ordinates of ABSOLUTETENSOR
        #
        abso_trans = absoluteTensor[:,:,0:self.translation_dim]
        abso_size = absoluteTensor[:,:,self.translation_dim:self.translation_dim+self.size_dim]
        abso_angle = absoluteTensor[:,:,self.bbox_dim-self.angle_dim:self.bbox_dim]
        abso_other = absoluteTensor[:,:,self.bbox_dim:]
        rela_trans = denoise_out[:,:,0:self.translation_dim]
        rela_size = denoise_out[:,:,self.translation_dim:self.translation_dim+self.size_dim]
        rela_angle = denoise_out[:,:,self.bbox_dim-self.angle_dim:self.bbox_dim]
        rela_other = denoise_out[:,:,self.bbox_dim:]
        new_trans = 0
        new_size = 0
        new_angle = 0
        new_other = abso_other + rela_other

        #scale
        if self.relativeScale:
            if self.relativeScaleMethod == "minus":
                new_size = abso_size + rela_size #add the x_recon's scale with denoise_out's scale
            if self.relativeScaleMethod == "divide":
                new_size = abso_size * rela_size #mutiply the x_recon's scale with denoise_out's scale
        else:
            new_size = rela_size #replace the x_recon's scale with denoise_out's scale
            
        if self.relativeScl_Ori_Trn:
            new_trans = abso_size * rela_trans #multiplies the denoise_out's translation with absoluteTensor's scale
        else:
            new_trans = rela_trans

        #rotate
        new_angle = F.normalize(self.simple_Norm_Minus(abso_angle, -rela_angle), dim=-1)
        
        new_trans = new_trans.reshape((self.batchsz, self.maxObj, 1, self.translation_dim))
        cs = abso_angle[:,:,:1].reshape((self.batchsz,-1,1,1))
        sn = abso_angle[:,:,1:].reshape((self.batchsz,-1,1,1))
        rotateX = torch.cat([cs, torch.zeros(cs.shape, device=cs.device), sn], axis = -1)
        rotateX.shape    (batchsz=128) : (maxObj=12) : (spatial_dim_hori=1) : (spatial_dim_vert=3)  #vertical vector in spatial space
        rotateY = torch.cat([torch.zeros(cs.shape, device=cs.device), torch.ones(cs.shape, device=cs.device), torch.zeros(cs.shape, device=cs.device)], axis = -1)
        rotateZ = torch.cat([-sn, torch.zeros(cs.shape, device=cs.device), cs], axis = -1)
        new_trans = (torch.cat([rotateX, rotateY, rotateZ], axis = -2) * new_trans).sum(axis=-1)
        #rotate.shape    (batchsz=128) : (src_dim=maxObj=12) : (spatial_dim_hori=3) : (spatial_dim_vert=3)  #3*3 matrix in spatial space
        #new_trans.shape (batchsz=128) : (src_dim=maxObj=12) : (spatial_dim_hori=3) : (spatial_dim_vert=1)  #horizontal vector in spatial space
        new_trans = new_trans.reshape((self.batchsz,self.maxObj,3))

        #translate
        new_trans = abso_trans + new_trans

        return torch.cat([new_trans, new_size, new_angle, new_other], axis=-1)
"""

class relaYL():
    def __init__(self, config):
        #all kinds of dimensions here
        self.batchsz = config.get("batchsz", 4)

        #................basic length of the data structure...................
            #.........absoluteTensor.....................
        self.objectness_dim = config.get("objectness_dim", 0) #exist or not, if objectness_dim=0, 
                                            #then we use the last digit of class_dim (or "end_label") as the objectness
        self.class_dim = config.get("class_dim", 25)
        self.translation_dim = config.get("translation_dim", 3)
        self.size_dim = config.get("size_dim", 3)
        self.angle_dim = config.get("angle_dim", 2)
        self.bbox_dim = self.translation_dim + self.size_dim + self.angle_dim
        self.objfeat_dim = config.get("objfeat_dim", 0)
        self.point_dim = config.get("point_dim", 65)
        self.maxObj = config.get("sample_num_points", 21)
        
        if self.point_dim != self.bbox_dim+self.class_dim+self.objectness_dim+self.objfeat_dim:
            raise NotImplementedError

            #.........wallTensor.....................
        self.process_wall = config.get("process_wall", False)
        self.wall_translation_dim = config.get("wall_translation_dim", 2)
        self.wall_norm_dim = config.get("wall_norm_dim",2)
        self.wall_dim = self.wall_translation_dim + self.wall_norm_dim
        self.maxWall = config.get("maxWall", 16)

            #.........windoorTensor.....................
        self.process_windoor = config.get("process_windoor", False)
        self.windoor_translation_dim = config.get("windoor_translation_dim", 3)
        self.windoor_scale_dim = config.get("windoor_scale_dim", 2)
        self.windoor_norm_dim = config.get("windoor_norm_dim", 2)
        self.windoor_dim = self.windoor_translation_dim + self.windoor_scale_dim + self.windoor_norm_dim
        self.maxWindoor = config.get("maxWindoor",8)

        #^^^^^^^^^^^^^^^^^^^^^basic length of the data structure^^^^^^^^^^^^^^^^^^^

        #.....................calculating process..................................
            #.............distance()
        self.relativeTranslation = config.get("relativeTranslation", True)
                #the form of relative translation
        self.relativeOrien_Trans = config.get("relativeOrien_Trans", True)
        self.relativeScl_Ori_Trn = config.get("relativeScl_Ori_Trn", False)

        self.relativeOrientation = config.get("relativeOrientation", True)
                #the form of relative angle: of course, their difference (or their rotating matrix, it's up to "self.angle_dim")

        self.relativeScale       = config.get("relativeScale", False)
                #the form of relative scale
        self.relativeScaleMethod = config.get("relativeScaleMethod", "divide")

                #the form of relative class labels ?????
        self.relativeClassLabels = config.get("relativeClassLabels", True)
        
                #the form of relative object feature ???  this could even become an interesting topic
        self.relativeObjFeatures = config.get("relativeObjFeatures", False)
        
                #the way to add these relative values together
        self.weightForm = config.get("weightForm", "exp-dis")
        self.alpha = config.get("diminish_coefficient", 1.0)
        self.nearestNum = config.get("nearestNum", 1)

            #.............wall()
        self.relative_wall_translation = config.get("relative_wall_translation", True)
        self.relative_wall_orientation = config.get("relative_wall_orientation", True)
        self.relative_wall_scale = config.get("relative_wall_scale", True)
        self.independent_wall = config.get("independent_wall", False)
        
            #.............windoor()
        self.relative_windoor_translation = config.get("relative_windoor_translation", True)
        self.relative_windoor_orientation = config.get("relative_windoor_orientation", True)
        self.relative_windoor_scale = config.get("relative_windoor_scale", True)
        self.independent_windoor = config.get("independent_windoor", False)

        if self.angle_dim != 2 or self.wall_norm_dim != 2 or self.windoor_norm_dim != 2:
            raise NotImplementedError

        if visual:
            self.visualizer = plotGeneral(4,self.batchsz,self.maxObj) #translating, translated-rotating, translated-rotated-scaling, selecting
        pass
    
    @staticmethod
    def norm_Minus(normA, normB): #[cosA, sinA]  [cosB, sinB]
        if prt:
            print("normA")
            print(normA.shape)
            print("normB")
            print(normB.shape)
        cs = (normA * normB).sum(axis=-1)[:,:,:,None]#.reshape((normA.shape[0], max(normA.shape[1],normB.shape[1]), max(normA.shape[2],normB.shape[2]), 1))  #cos(A-B) = cosBcosA + sinBsinA
        na = normA.reshape((-1,2))
        if prt:
            print("na")
            print(na.shape)
        nA = torch.cat([na[:,1:], -na[:,:1]], axis=-1)
        if prt:
            print("nA")
            print(nA.shape)
        nA = nA.reshape(normA.shape)
        if prt:
            print("nA")
            print(nA.shape)
        sn = (normB * nA).sum(axis=-1)[:,:,:,None]#.reshape((normA.shape[0], max(normA.shape[1],normB.shape[1]), max(normA.shape[2],normB.shape[2]), 1))     #sin(A-B) =-sinBcosA + cosBsinA
        return torch.cat([cs,sn],axis=-1)

    @staticmethod
    def simple_Norm_Minus(normA, normB):
        cs = (normA * normB).sum(axis=-1)[:,:,None]#.reshape((normA.shape,1))
        na = normA.reshape((-1,2))
        nA = torch.cat([na[:,1:], -na[:,:1]], axis=-1).reshape(normA.shape)
        sn = (normB * nA).sum(axis=-1)[:,:,None]#.reshape((normA.shape,1))
        return torch.cat([cs,sn],axis=-1)

    @staticmethod
    def theta2norm(theta):
        return torch.cat([torch.cos(theta), torch.sin(theta)], axis=-1)
    
    @staticmethod
    def theta_Minus(thetaA, thetaB):
        return thetaA - thetaB #cast from -pi to pi

    def windoor(self, absoluteTensor, windoorTensor):
        # batchsz = 128 : maxObj = 12 : bbox_dim = 8         absolutionTensor.dim = -1:
        # (trans_x, ..y, ..z, scale_x, ..y, ..z, norm_Z(cosA), norm_X(sinA) ) #Z before X in norm

        # batchsz = 128 : maxW = 8 : point_dim = 8         windoorTensor.dim = -1:
        # (midTranslation_dim=3)+(scale_dim=3)+(innerNorm_dim=2) 

        xx = absoluteTensor.reshape((self.batchsz,self.maxObj,1,-1))
        if prt:
            print("xx")
            print(xx.shape)

        yy = windoorTensor.reshape((self.batchsz,1,-1,self.windoor_dim))
        if prt:
            print("yy")
            print(yy.shape)

        if not self.relative_wall_translation:
            raise NotImplementedError
        #self.relative_wall_translation == True: #wall translation minus object translation

        tZ = yy[:,:,:,self.windoor_translation_dim-1:self.windoor_translation_dim] - xx[:,:,:,self.translation_dim-1:self.translation_dim]
        tX = yy[:,:,:,0:1] - xx[:,:,:,0:1]
        swaptCo_ord = torch.cat([tZ, tX], axis=-1)
        windoorLength = yy[:,:,:,self.windoor_translation_dim:self.windoor_translation_dim+1]

        dis = (swaptCo_ord**2).sum(axis=-1)
        dis = dis.reshape((self.batchsz, xx.shape[1], yy.shape[2]))
        if prt:
            print("dis")
            print(dis.shape)
            #print(dis.tolist())

        if visual: #[windoorMid_x, ...y, ...z, windoorScale_x, ...y, ...z, windoowNorm_z, ...x, rotate_z, ...x] state = 0: translating,
            #windoorNorm = (windoorNorm_z, windoorNorm_x) #first z then x
            #windoorVector = ( windoorNorm_z * windoorLen, - windoorNorm_x * windoorLen ) #first x then z
            #swapWindoorVector = ( - windoorNorm_x * windoorLen, windoorNorm_z * windoorLen) #first z then x
            tCo_ord = yy[:,:,:,:self.windoor_translation_dim] - xx[:,:,:,:self.windoor_translation_dim]
            yys = yy[:,:,:,self.windoor_translation_dim:].repeat((1,self.maxObj,1,1))
            ct = torch.cat([tCo_ord, yys], axis=-1)
            self.visualizer.plotWindoors(ct,0)
            pass

        if not self.relative_wall_orientation:
            raise NotImplementedError
        #self.relative_wall_orientation == True: #rotate with the object angle
        
        #theta in source object's co-ordinates        
        if self.angle_dim == 2:
            swaprtCo_ord = self.norm_Minus(swaptCo_ord, xx[:,:,:,self.bbox_dim-self.angle_dim: self.bbox_dim])
        else: #self.angle_dim == 1
            raise NotImplementedError #still buggy!!!!
            tCo_ord = torch.arctan2(relX,relZ).reshape((self.batchsz,self.maxObj,-1,1)) - xx[:,:,:,self.bbox_dim-self.angle_dim:self.bbox_dim]

        swaprtCo_ord = swaprtCo_ord.reshape((self.batchsz, xx.shape[1], yy.shape[2], self.angle_dim))
        if prt:
            print("swaprtCo_ord")
            print(swaprtCo_ord.shape)
            #print(swaprtCo_ord.tolist())

        relCo_ord = torch.cat([swaprtCo_ord[:,:,:,-1:], swaprtCo_ord[:,:,:,:1]], axis = -1)

        if self.angle_dim == 2: 
            if self.windoor_norm_dim == 2:
                windoor_norm = yy[:,:,:,self.windoor_dim-self.windoor_norm_dim:self.windoor_dim]
            else:
                raise NotImplementedError
                windoor_norm = self.theta2norm(yy[:,:,:,self.windoor_dim-self.windoor_norm_dim:self.windoor_dim])
            rWNV = self.norm_Minus(windoor_norm, xx[:,:,:,self.bbox_dim-self.angle_dim:self.bbox_dim])
        else: #self.angle_dim = 1
            raise NotImplementedError
            if self.windoor_norm_dim == 2:
                raise NotImplementedError
                #arctan2
            else:
                windoor_angle = yy[:,:,:,self.windoor_dim-self.windoor_norm_dim:self.windoor_dim]
            
            rWNV = (windoor_angle - xx[:,:,:,self.bbox_dim-self.angle_dim:self.bbox_dim])
        
        rWNV = rWNV.reshape((self.batchsz, xx.shape[1], yy.shape[2], self.angle_dim))
        if prt:
            print("rWNV")
            print(rWNV.shape)
            #print(rWNV.tolist())

        relTheta = rWNV

        if visual: #[windoorMid_x, ...y, ...z, windoorScale_x, ...y, ...z, windoowNorm_z, ...x, rotate_z, ...x] state = 1: translated - rotating
            ct = torch.cat([relCo_ord[:,:,:,:1], yy[:,:,:,1:2]-xx[:,:,:,1:2], relCo_ord[:,:,:,1:2], yys[:,:,:,:self.windoor_scale_dim], rWNV],axis=-1)
            self.visualizer.plotWindoors(ct,1)
            pass

        if self.relative_windoor_scale:
            #self.relative_wall_scale == True: #scale the relative translation and the relative norm-vector
            swapScale = torch.cat([xx[:,:,:,self.translation_dim+self.size_dim-1:self.translation_dim+self.size_dim], xx[:,:,:,self.translation_dim:self.translation_dim+1]], axis=-1) #[scale_z, scale_x]
            swaprWV = torch.cat([rWNV[:,:,:,:1],-rWNV[:,:,:,1:]], axis=-1)
            swapsrWV = swaprWV / swapScale
            lengthScale = ((swapsrWV**2).sum(axis=-1)**0.5).reshape((self.batchsz,-1,self.maxWindoor,1))
            windoorLength = windoorLength * lengthScale

            swapsrtCo_ord = swaprtCo_ord / swapScale 
            srWNV = rWNV * swapScale
            relTheta = F.normalize(srWNV, dim=-1)
            if prt:
                print("relTheta")
                print(relTheta.shape)
            relCo_ord = torch.cat([swapsrtCo_ord[:,:,:,-1:], swapsrtCo_ord[:,:,:,:1]], axis = -1)
            dis = (swapsrtCo_ord**2).sum(axis=-1)
            dis = dis.reshape((self.batchsz, xx.shape[1], yy.shape[2]))

        if visual: #[windoorMid_x, ...y, ...z, windoorScale_x, ...y, ...z, windoowNorm_z, ...x, rotate_z, ...x] state = 2: translated - rotated - scaling
            if self.relative_windoor_scale:
                ct = torch.cat([relCo_ord[:,:,:,:1], yy[:,:,:,1:2]-xx[:,:,:,1:2], relCo_ord[:,:,:,1:2], windoorLength, yys[:,:,:,1:2], relTheta], axis = -1)
                self.visualizer.plotWindoors(ct,2)
            else:
                self.visualizer.plotWindoors(ct,2)

        #then we use the mixing step--------------------------dis, relCo_ord, relTheta-------------------------

        mins, argmins = dis.min(axis=-1)
        if prt:
            print("mins")
            print(mins.shape)
            #print(mins.tolist())
            print("argmins")
            print(argmins.shape)
            #print(argmins.tolist())
            #print(torch.arange(batchsz).reshape((-1,1)).repeat(1, argmins.shape[1]).shape)
            #print(torch.arange(argmins.shape[1]).reshape((1,-1)).repeat(batchsz, 1).shape)

        id0 = torch.arange(self.batchsz  ).reshape((-1,1,1)).repeat(1,self.maxObj,self.angle_dim)
        id1 = torch.arange(self.maxObj   ).reshape((1,-1,1)).repeat(self.batchsz,1,self.angle_dim)
        id2 = argmins.reshape((self.batchsz,self.maxObj,1)).repeat(1,1,self.angle_dim)
        id3 = torch.arange(self.angle_dim).reshape((1,1,-1)).repeat(self.batchsz,self.maxObj,1)
        #thetamin.shape = (batchsz=128) : (maxObj=12) : (angle_dim=1/2)

        thetamin = relTheta[id0, id1, id2, id3]
        if prt:
            print("thetamin")
            print(thetamin.shape)
            #print(thetamin.tolist())

        comin = relCo_ord[id0, id1, id2, id3]
        if prt:
            print("comin")
            print(comin.shape)
            #print(comin.tolist())
        
        scl = torch.cat( [windoorLength, yy[:,:,:,4:5].repeat((1,self.maxObj,1,1))], axis=-1 )
        sclmin = scl[id0,id1,id2,id3]
        
        if visual: #[windoorMid_x, ...y, ...z, windoorScale_x, ...y, ...z, windoowNorm_z, ...x, rotate_z, ...x] state = 3: translated - rotated - scaled - selecting
            fd = ct.shape[-1]
            id00 = torch.arange(self.batchsz  ).reshape((-1,1,1)).repeat(1,self.maxObj,fd)
            id11 = torch.arange(self.maxObj   ).reshape((1,-1,1)).repeat(self.batchsz,1,fd)
            id22 = argmins.reshape((self.batchsz,self.maxObj,1)).repeat(1,1,fd)
            id33 = torch.arange(fd).reshape((1,1,-1)).repeat(self.batchsz,self.maxObj,1)
            ctt = ct[id00,id11,id22,id33]
            self.visualizer.plotWindoors(ctt,3)
            pass

        if self.independent_windoor:
            kk = self.windoor_dim
            id00 = torch.arange(self.batchsz  ).reshape((-1,1,1)).repeat(1,self.maxObj,kk)
            id11 = torch.arange(self.maxObj   ).reshape((1,-1,1)).repeat(self.batchsz,1,kk)
            id22 = argmins.reshape((self.batchsz,self.maxObj,1)).repeat(1,1,kk)
            id33 = torch.arange(kk).reshape((1,1,-1)).repeat(self.batchsz,self.maxObj,1)
            #thetamin.shape = (batchsz=128) : (maxObj=12) : (angle_dim=1/2)
            wt = windoorTensor.reshape((self.batchsz,1,-1,self.windoor_dim)).repeat(1,self.maxObj,1,1)
            minWindoor = wt[id00, id11, id22, id33]
            return minWindoor

        resultTensor = torch.cat([comin.reshape((self.batchsz,-1,2)), sclmin.reshape((self.batchsz,-1,2)), thetamin.reshape((self.batchsz,-1,self.angle_dim))],axis = -1)
        
        # batchsz = 128 : maxObj = 12 : point_dim = 5/6       resultTensor.dim = -1:
        # co_ord(2) + length(1) + height(1) + theta(1)/dire(2)    #this result is not enough, there still other things to be done
        return resultTensor   #resultTensor = absoluteTensor

    def wall(self, absoluteTensor, wallTensor):
        # batchsz = 128 : maxObj = 12 : bbox_dim = 8         absolutionTensor.dim = -1:
        # (trans_x, trans_y, trans_z, scale_x, scale_y, scale_z, norm_Z(cosA), norm_X(sinA) )

        # batchsz = 128 : maxW = 16 : point_dim = 4         wallTensor.dim = -1:
        # point_x, point_z, innerNorm_x, innerNorm_z 

        wal = torch.cat([wallTensor[:,-1:], wallTensor[:,:-1]], axis = -2)
        if prt:
            print("wal")
            #print(wal.tolist())
            print(wal.shape)
        wallStart = wal[:,:,:self.wall_translation_dim]
        wallNorm = wal[:,:,self.wall_translation_dim:self.wall_translation_dim+self.wall_norm_dim]
        if prt:
            print("wallNorm")
            print(wallNorm.shape)
            #print(wallNorm.tolist())
        
        xx = absoluteTensor.reshape((self.batchsz,self.maxObj,1,-1))
        if prt:
            print("xx")
            print(xx.shape)

        WS = wallStart.reshape((self.batchsz,1,self.maxWall,self.wall_translation_dim))
        if prt:
            print("WS")
            print(WS.shape)

        #the problem right now is to re-organize this stuff

        #we have what?
            #Under World Co-ordinates
                #Object-Translation     Object-size     Object-Normalization
                #Wall-Center, Wall-Normalization, Wall-Length

            #finally, we can calculate the distance, shift, bound, orientation under "this co-ordinates"

        #during the process, we maintains: relative wallCenter, relative wallNormVecter, relative wallTangVector
        #at the end, we calculate: distance and orientation, shift&bound with the wC, wNV, wTV

        if not self.relative_wall_translation:
            raise NotImplementedError
        #self.relative_wall_translation == True: #wall translation minus object translation

        tWS = torch.cat([WS[:,:,:,0:1] - xx[:,:,:,0:1], WS[:,:,:,self.wall_translation_dim-1:self.wall_translation_dim] - xx[:,:,:,self.translation_dim-1:self.translation_dim] ], axis=-1)
        if prt:
            print("tWS")
            print(tWS.shape)
            #print(tWS.tolist())
        swaptWS = torch.cat([tWS[:,:,:,1:2], tWS[:,:,:,0:1]], axis=-1)
        
        dis = (swaptWS * wallNorm.reshape((self.batchsz,1,-1,self.wall_norm_dim))).sum(axis=-1)
        if prt:
            print("dis")
            print(dis.shape)
            #print(dis.tolist())

        #swaptWS[0] * wallNorm[1] - swaptWS[1] * wallNorm[0]
        aaa = self.norm_Minus(swaptWS, wallNorm.reshape((self.batchsz,1,-1,self.wall_norm_dim)))[:,:,:,1] 

        #一方面，无论是否进行旋转，dis的值都不会变，（如果需要进行仿射，则dis还需要重置）
        #另外，无论是否需要进行旋转和仿射，实际上shift和bound之间的关系都应当是确定的，因此我们在这里先把shift和bound定下来了

        if visual:  #[wallStart_x, wallStart_z, wallNorm_z, wallNorm_x] state = 0: translating,
            wallN = wallNorm.reshape((self.batchsz,1,-1,self.wall_norm_dim)).repeat((1,self.maxObj,1,1))
            ct = torch.cat([tWS, wallN], axis=-1)
            self.visualizer.plotWalls(ct,0)

        if not self.relative_wall_orientation:
            raise NotImplementedError
        #self.relative_wall_orientation == True: #rotate with the object angle
        
        #wNV rotate with -angle
        if self.angle_dim == 1:
            raise NotImplementedError
            rWNV = torch.arctan2( wallNorm[:,:,1:2] , wallNorm[:,:,0:1] ).reshape((batchsz,1,-1,self.angle_dim)) - xx[:,:,:,self.bbox_dim-self.angle_dim:self.bbox_dim].reshape((batchsz,-1,1,self.angle_dim))
        else:
            rWNV = self.norm_Minus(wallNorm.reshape(self.batchsz,1,-1,self.wall_norm_dim), xx[:,:,:,self.bbox_dim-self.angle_dim:self.bbox_dim])
        if prt:#or True:
            print("rWNV")
            print(rWNV.shape)
            #print(rWNV.tolist())

        relTheta = rWNV

        if visual:  #[wallStart_x, wallStart_z, wallNorm_z, wallNorm_x] state = 1: translated - rotating 
            swaprtWS = self.norm_Minus(swaptWS, xx[:,:,:,self.bbox_dim-self.angle_dim:self.bbox_dim])
            print("swaprtWS")
            print(swaprtWS.shape)
            rtWS = torch.cat([swaprtWS[:,:,:,1:2], swaprtWS[:,:,:,0:1]], axis=-1)
            print("rtWS")
            print(rtWS.shape)
            ct = torch.cat([rtWS, rWNV], axis=-1)
            self.visualizer.plotWalls(ct,1)
            pass

        if self.relative_wall_scale:
            #self.relative_wall_scale == True: #scale the relative translation and the relative norm-vector
            #wC rotate with -angle (we should form a matrix for it)
            if self.angle_dim == 1:
                raise NotImplementedError
            else:
                swaprtWS = self.norm_Minus(swaptWS,xx[:,:,:,self.bbox_dim-self.angle_dim:self.bbox_dim])
            
            if prt:
                print("swaprtWS")
                print(swaprtWS.shape)

            swapScale = torch.cat([xx[:,:,:,self.translation_dim+self.size_dim-1:self.translation_dim+self.size_dim], xx[:,:,:,self.translation_dim:self.translation_dim+1]], axis=-1) #[scale_z, scale_x]
            swapsrtWS = swaprtWS / swapScale 
            if prt:
                print("swapsrtWS")
                print(swapsrtWS.shape)

            srWNV = rWNV * swapScale
            relTheta = F.normalize(srWNV, dim=-1)
            dis = (swapsrtWS * relTheta).sum(axis=-1)
            aaa = self.norm_Minus(swapsrtWS, relTheta)[:,:,:,1] 
        
        if visual:  #[wallStart_x, wallStart_z, wallNorm_z, wallNorm_x] state = 2: translated - rotated - scaling 
            if self.relative_wall_scale:
                srtWS = torch.cat([swapsrtWS[:,:,:,1:2], swapsrtWS[:,:,:,0:1]], axis=-1)
                ct = torch.cat([srtWS, relTheta], axis=-1)
                self.visualizer.plotWalls(ct,2)
            else:
                ct = torch.cat([rtWS, rWNV], axis=-1)
                self.visualizer.plotWalls(ct,2)
            
        #then we use the mixing step----------------(dis: (batchsz, maxObj, maxW, 1), relTheta: (batchsz, maxObj, maxW, 2?))------------------------------

        swaptWE = torch.cat([swaptWS[:,:,1:,:], swaptWS[:,:,:1,:]], axis=-2)
        bbb = self.norm_Minus(swaptWE, wallNorm.reshape((self.batchsz,1,-1,self.wall_norm_dim)))[:,:,:,1] 
        aaabbb = aaa * bbb
        cond = aaabbb < torch.zeros_like(aaabbb)
        #cond = lengthSquare < lenSquare.reshape((self.batchsz,1,-1))
        if prt:
            print("cond")
            print(cond.shape)
            #print(cond.tolist())
        res = torch.ones_like(dis) * 10000
        res[cond] = dis[cond]
        if prt:
            print("res")
            print(res.shape)

        mins, argmins = abs(res).min(axis=-1)
        if prt:
            print("mins")
            print(mins.shape)
            #print(mins.tolist())
            print(argmins.shape)
            #print(argmins.tolist())
            #print(torch.arange(self.batchsz).reshape((-1,1)).repeat(1, argmins.shape[1]).shape)
            #print(torch.arange(argmins.shape[1]).reshape((1,-1)).repeat(self.batchsz, 1).shape)

        #argmins.shape = (batchsz=128) : (maxObj=12) : (result=1)
        #index tensor is: for a mother tensor with N dimensions, we need N index tensor with same shape
        #mother[ indexTensor_0, indexTensor_1, ...... indexTensor_N-1]
        #while the shape of these indexTensor turns out to be the shape of resultTensor
        #the value across each indexTensor turns out to be the index of each element
        #now lets go for our own case

        #relTheta.shape = (batchsz=128) : (maxObj=12) : (maxWall=16) : (angle_dim=1/2)
        #                00..11...127127  0123...0123.. argmins repeat 01010101010101
        #                  (128*12*1/2) : (128*12*1/2) : (128*12*1/2) : (128*12*1/2)

        id0 = torch.arange(self.batchsz  ).reshape((-1,1,1)).repeat(1,self.maxObj,self.angle_dim)
        id1 = torch.arange(self.maxObj   ).reshape((1,-1,1)).repeat(self.batchsz,1,self.angle_dim)
        id2 = argmins.reshape((self.batchsz,self.maxObj,1)).repeat(1,1,self.angle_dim)
        id3 = torch.arange(self.angle_dim).reshape((1,1,-1)).repeat(self.batchsz,self.maxObj,1)
        #thetamin.shape = (batchsz=128) : (maxObj=12) : (angle_dim=1/2)

        thetamin = relTheta[id0, id1, id2, id3]
        if prt:
            print("thetamin")
            print(thetamin.shape)
            #print(thetamin.tolist())

        if visual:  #[wallStart_x, wallStart_z, wallNorm_z, wallNorm_x] state = 3: translated - rotated - scaled - selecting
            argminsNext = argmins + torch.ones_like(argmins)
            id22 = argminsNext.reshape((self.batchsz,self.maxObj,1)).repeat(1,1,self.angle_dim)
            if self.relative_wall_scale:
                WSmin =srtWS[id0, id1, id2, id3]
                WSminNext = srtWS[id0, id1, id22, id3]
            else:
                WSmin = rtWS[id0, id1, id2, id3]
                WSminNext = rtWS[id0, id1, id22, id3]
            print("argmins")
            print(argmins.shape)
            print("WSmin")
            print(WSmin.shape)
            WSmin_next = torch.cat([WSmin.reshape(self.batchsz,self.maxObj,1,-1), WSminNext.reshape(self.batchsz,self.maxObj,1,-1)], axis=-2)
            print("WSmin_next")
            print(WSmin_next.shape)
            ct = WSmin_next #torch.cat([WSmin_next, thetamin], axis=-1)
            self.visualizer.plotWalls(ct,3)

        if self.independent_wall:
            kk = self.wall_dim + self.wall_translation_dim
            id00 = torch.arange(self.batchsz  ).reshape((-1,1,1)).repeat(1,self.maxObj,kk)
            id11 = torch.arange(self.maxObj   ).reshape((1,-1,1)).repeat(self.batchsz,1,kk)
            id22 = argmins.reshape((self.batchsz,self.maxObj,1)).repeat(1,1,kk)
            id33 = torch.arange(kk).reshape((1,1,-1)).repeat(self.batchsz,self.maxObj,1)
            #thetamin.shape = (batchsz=128) : (maxObj=12) : (angle_dim=1/2)
            wallEnd = torch.cat([wallStart[:,1:],wallStart[:,:1]],axis=-2)
            fullWall = torch.cat([wallStart, wallEnd, wallNorm],axis=-1).reshape((self.batchsz,1,self.maxWall,-1))
            fullWall = fullWall.repeat(1,self.maxObj,1,1)
            minWall = fullWall[id00, id11, id22, id33]
            return minWall
            pass

        resultTensor = torch.cat([mins.reshape((self.batchsz,-1,1)),thetamin.reshape((self.batchsz,-1,self.angle_dim))],axis = -1) #,argmins.reshape((batchsz,-1,1))
        #print(resultTensor.shape)

        # batchsz = 128 : maxObj = 12 : point_dim = 2/3         resultTensor.dim = -1:
        # min_Distance2 + Orientation(1) / Norm(2)                       #resultTensor = absoluteTensor
        return resultTensor

    def distance(self, absoluteTensor, square=True):
        # batchsz = 128 : maxObj = 12 : point_dim = 6?         absolutionTensor.dim = -1:
        # (translation_dim=3)+(size_dim=3)+(angle_dim=2) + (class_dim = 2?) + (objfeat_dim = 32)
        # (             bbox_dim   =   8    cosA, sinA )
        # (                          point_dim     =     6?                                    )
        xx = absoluteTensor.reshape((self.batchsz,-1,1,self.point_dim))
        if prt:
            print("xx")
            print(xx.shape)
        #xx.shape    (batchsz=128) : (src_dim=maxObj=12) : (dst_dim=1) : (translation_dim + angle_dim=4)

        yy = absoluteTensor.reshape((self.batchsz,1,-1,self.point_dim))
        if prt:
            print("yy")
            print(yy.shape)
        #yy.shape    (batchsz=128) : (src_dim=1) : (dst_dim=maxObj=12)  : (translation_dim + angle_dim=4)

        rel = (yy[:,:,:,:self.translation_dim] - xx[:,:,:,:self.translation_dim]).reshape(self.batchsz,self.maxObj,-1,1,self.translation_dim)
        if prt:
            print("rel")
            print(rel.shape)
            #print(rel.tolist())
        #rel.shape    (batchsz=128) : (src_dim=maxObj=12) : (dst_dim=maxObj=12) : (spatial_dim_hori=1) : (spatial_dim_vert=3)  #vertical vector in spatial space

        if visual:  #[3 + 3 + 2] state = 0: translating,
            yys = yy[:,:,:,self.translation_dim:self.bbox_dim].repeat((1,self.maxObj,1,1))
            rels = rel.reshape((self.batchsz,self.maxObj,-1,self.translation_dim))
            ct = torch.cat([rels, yys], axis=-1)
            self.visualizer.plotObjs(ct,0)
            pass

        if self.angle_dim == 1:
            relTheta = xx[:,:,:,self.bbox_dim-self.angle_dim:self.bbox_dim] - yy[:,:,:,self.bbox_dim-self.angle_dim:self.bbox_dim]
            #relTheta.shape (batchsz=128) : (src_dim=maxObj=12) : (dst_dim=maxObj=12) : (angle_dim=1)
            cs = torch.cos(xx[:,:,:,self.bbox_dim-self.angle_dim:self.bbox_dim]).reshape((self.batchsz,-1,1,1,1))
            #cs.shape    (batchsz=128) : (src_dim=maxObj=12) : (dst_dim=1) : (spatial_dim_hori=1) : (spatial_dim_vert=1)  #scaler in spatial space
            sn = torch.sin(xx[:,:,:,self.bbox_dim-self.angle_dim:self.bbox_dim]).reshape((self.batchsz,-1,1,1,1))
        else:
            relTheta = self.norm_Minus(yy[:,:,:,self.bbox_dim-self.angle_dim:self.bbox_dim], xx[:,:,:,self.bbox_dim-self.angle_dim:self.bbox_dim])
            cs = xx[:,:,:,self.bbox_dim-self.angle_dim:self.bbox_dim-self.angle_dim+1].reshape((self.batchsz,-1,1,1,1))
            sn = xx[:,:,:,self.bbox_dim-1:self.bbox_dim].reshape((self.batchsz,-1,1,1,1))
        #print(cs.shape)
        #print(sn.shape)

        rotateX = torch.cat([cs, torch.zeros(cs.shape, device=cs.device),-sn], axis = -1)
        #rotateX.shape    (batchsz=128) : (src_dim=maxObj=12) : (dst_dim=1) : (spatial_dim_hori=1) : (spatial_dim_vert=3)  #vertical vector in spatial space
        rotateY = torch.cat([torch.zeros(cs.shape, device=cs.device), torch.ones(cs.shape, device=cs.device), torch.zeros(cs.shape, device=cs.device)], axis = -1)
        rotateZ = torch.cat([sn, torch.zeros(cs.shape, device=cs.device), cs], axis = -1)
        #print(rotateZ.shape)

        rotate = torch.cat([rotateX, rotateY, rotateZ], axis = -2)
        #rotate.shape    (batchsz=128) : (src_dim=maxObj=12) : (dst_dim=1) : (spatial_dim_hori=3) : (spatial_dim_vert=3)  #3*3 matrix in spatial space
        #print(rotate.shape)
        #print(rel.shape)
        newRel = (rotate * rel).sum(axis=-1)
        #newRel.shape    (batchsz=128) : (src_dim=maxObj=12) : (dst_dim=maxObj=12) : (spatial_dim_hori=3) : (spatial_dim_vert=1)  #horizontal vector in spatial space
        newRel = newRel.reshape((self.batchsz,-1,self.maxObj,3))
        #newRel.shape    (batchsz=128) : (src_dim=maxObj=12) : (dst_dim=maxObj=12) : (translation_dim=3) #flatten spatial space

        if prt:
            print("relTheta")
            print(relTheta.shape)
            #print(relTheta.tolist())

        dis2 = (rel ** 2).sum(axis=-1).reshape((self.batchsz,self.maxObj,-1))
        if prt:
            print("dis2")
            print(dis2.shape)
            #print(dis2.tolist())
        #dis2.shape    (batchsz=128) : (src_dim=maxObj=12) : (dst_dim=maxObj=12)

        if visual:  #[3 + 3 + 2] state = 1: translated - rotating 
            yys = yy[:,:,:,self.translation_dim:self.translation_dim+self.size_dim].repeat((1,self.maxObj,1,1))
            ct = torch.cat([newRel, yys, relTheta], axis=-1)
            self.visualizer.plotObjs(ct,1)
            pass

        expdis2 = torch.exp(-dis2)
        expdis2_0 = torch.zeros_like(expdis2)
        #expdis2.shape    (batchsz=128) : (src_dim=maxObj=12) : (dst_dim=maxObj=12)

        #ban掉那些class_dim末位是>0的那些index， 让他们的expdis2 = 0
        #absoluteTensor.shape = batchsz = 128 : maxObj = 12 : fullLength = 62
        
        end_label = absoluteTensor[:,:,self.bbox_dim+self.class_dim-1:self.bbox_dim+self.class_dim] #30???????

        #end_label.shape : (batchsz = 128) : (ambiguous_dim = maxObj = 12) : (label_dim = 1)
        cond = (end_label > torch.zeros_like(end_label)).reshape((self.batchsz, 1, self.maxObj))
        #cond.shape    (batchsz=128) : (src_dim=1) : (dst_dim=maxObj=12) 
        cond = cond.repeat((1, self.maxObj, 1))
        #cond.shape    (batchsz=128) : (src_dim=maxObj=12)  : (dst_dim=maxObj=12)
        expdis2[cond] = expdis2_0[cond]

        sumexpdis2 = expdis2.sum(axis=-1).reshape((self.batchsz,-1,1))
        #sumexpdis2.shape (batchsz=128) : (src_dim=maxObj=12)  : (dst_dim=1) # for each source, the destination of it are summed up 

        weights = (expdis2 / sumexpdis2).reshape((self.batchsz,self.maxObj,-1,1)) #(batchsz, 12, 12)
        #weights.shape    (batchsz=128) : (src_dim=maxObj=12)  : (dst_dim=maxObj=12) : (feat_dim=1) # to be broadcasted 

        #how to use this fucking weights
        #weights[i][j]  the j-th object weight to i-th  object

        #weights[0][0]      weights[0][1]       weights[0][2]=0     weights[0][3] 
        #weights[1][0]      weights[1][1]       weights[1][2]=0     weights[1][3]
        #weights[2][0]!=0   weights[2][1]!=0    weights[2][2]=0     weights[2][3]!=0
        #weights[3][0]      weights[3][1]       weights[3][2]=0     weights[3][3]

        #the problem is how to get the relative value of class_label and objfeats, these things can be controled by the config

        bb_sizes = absoluteTensor[:,:,self.translation_dim:self.translation_dim + self.size_dim]
        cl_class_labels = absoluteTensor[:,:,self.bbox_dim:self.bbox_dim+self.class_dim] #shape (batchsz=128) : (maxObj=12) : (feat_dim = 22)
        of_objfeats_32 = absoluteTensor[:,:,self.bbox_dim+self.class_dim:]

        xx_sizes = bb_sizes.reshape((self.batchsz,-1, 1, self.size_dim))
        yy_sizes = bb_sizes.reshape((self.batchsz, 1,-1, self.size_dim))

        if self.relativeScale:
            if self.relativeScaleMethod == "minus":
                relScale = (yy_sizes - xx_sizes).reshape((self.batchsz,self.maxObj,-1,self.size_dim))
                #relScale.shape    (batchsz=128) : (src_dim=maxObj=12)  : (dst_dim=maxObj=12) : (feat_dim = 3)
            if self.relativeScaleMethod == "divide":
                relScale = (yy_sizes / xx_sizes).reshape((self.batchsz,self.maxObj,-1,self.size_dim))
        else:
            relScale = yy_sizes.repeat((1,self.maxObj,1,1))

        if self.relativeScl_Ori_Trn:
            newRel = (newRel / xx_sizes).reshape((self.batchsz,-1,self.maxObj,3))

        if square:
            xx_class = cl_class_labels.reshape((self.batchsz,-1, 1, self.class_dim))
            yy_class = cl_class_labels.reshape((self.batchsz, 1,-1, self.class_dim))
            relClass = xx_class - yy_class
            resultTensor = torch.cat([newRel, relTheta, relClass], axis = -1)
            return resultTensor

        if visual:  #[3 + 3 + 2] state = 2: translated - rotated - scaling 
            ct = torch.cat([newRel, relScale, relTheta], axis=-1)
            self.visualizer.plotObjs(ct,2)
            pass

        relCL = cl_class_labels.reshape((self.batchsz,1,self.maxObj,-1)).repeat((1,self.maxObj,1,1)) #?????/
        relOF = of_objfeats_32.reshape((self.batchsz,1,self.maxObj,-1)).repeat((1,self.maxObj,1,1)) #?????/

        fullRel = torch.cat([newRel, relScale, relTheta, relCL, relOF], axis = -1)
        if prt:
            print("fullRel")
            print(fullRel.shape)
        #fullRel.shape    (batchsz=128) : (src_dim=maxObj=12)  : (dst_dim=maxObj=12) : (feat_dim = ???)

        #then we use the mixing step----------------(dis: (batchsz, maxObj, maxW, 1), relTheta: (batchsz, maxObj, maxW, 2?))------------------------------

        resultTensor = (fullRel * weights).sum(axis=-2).reshape((self.batchsz, self.maxObj, -1)) #print(resultTensor.shape)

        if visual:  #[wallStart_x, wallStart_z, wallNorm_z, wallNorm_x] state = 3: translated - rotated - scale - selecting
            #temporarily same as the state=2
            ct = torch.cat([newRel, relScale, relTheta], axis=-1)
            self.visualizer.plotObjs(ct,3)
            pass

        return resultTensor  #resultTensor = absoluteTensor
    
    def rela(self, absoluteTensor, wallTensor, windoorTensor):
        self.batchsz = absoluteTensor.shape[0]
        if prt:
            print("absoluteTensor.shape")
            print(absoluteTensor.shape)
            print("wallTensor.shape")
            print(wallTensor.shape)
        
        relativeWall = self.wall(absoluteTensor[:,:,:self.bbox_dim], wallTensor) if self.process_wall else None
        if prt and self.process_wall:
            print("relativeWall.shape")
            print(relativeWall.shape)
            print("windoorTensor.shape")
            print(windoorTensor.shape)

        relativeWindoor = self.windoor(absoluteTensor[:,:,:self.bbox_dim], windoorTensor) if self.process_windoor else None
        if prt and self.process_windoor:
            print("relativeWindoor.shape")
            print(relativeWindoor.shape)
        relativeObject = self.distance(absoluteTensor,False)
        if prt:
            print("relativeObject.shape")
            print(relativeObject.shape)

        if self.process_wall and self.process_windoor:
            relativeTensor = torch.cat([relativeObject, relativeWall, relativeWindoor], axis=-1)
        elif self.process_wall:
            relativeTensor = torch.cat([relativeObject, relativeWall], axis=-1)
        elif self.process_windoor:
            relativeTensor = torch.cat([relativeObject, relativeWindoor], axis=-1)
        else:
            relativeTensor = relativeObject
        if prt:
            print("relativeTensor.shape")
            print(relativeTensor.shape)
    
        end_label = absoluteTensor[:,:,self.bbox_dim+self.class_dim-1:self.bbox_dim+self.class_dim]
        #end_label.shape : (batchsz = 128) : (ambiguous_dim = maxObj = 12) : (label_dim = 1)
        cond = (end_label > torch.zeros_like(end_label))
        #cond.shape    (batchsz=128) : (src_dim=1) : (dst_dim=maxObj=12) 
        cond = cond.repeat((1, 1, relativeTensor.shape[-1]))
        #cond.shape    (batchsz=128) : (src_dim=maxObj=12)  : (dst_dim=maxObj=12)
        if prt:
            print("cond.shape")
            print(cond.shape)
        relativeZero = torch.zeros_like(relativeTensor)
        relativeTensor[cond] = relativeZero[cond]
        #别忘了ban掉那些class_dim末位是>0的那些index， 让他们的行掉为0
        if prt:
            print(relativeTensor.dtype)
        return relativeTensor

    def newRela(self, absoluteTensor, wallTensor=None, windoorTensor=None):
        self.batchsz = absoluteTensor.shape[0]
        end_label = absoluteTensor[:,:,self.bbox_dim+self.class_dim-1:self.bbox_dim+self.class_dim]
        #end_label.shape : (batchsz = 128) : (ambiguous_dim = maxObj = 12) : (label_dim = 1)
        con = (end_label > torch.zeros_like(end_label))
        #cond.shape    (batchsz=128) : (src_dim=1) : (dst_dim=maxObj=12) 
        
        if self.process_wall:
            relativeWall = self.wall(absoluteTensor[:,:,:self.bbox_dim], wallTensor)
            cond = con.repeat((1, 1, relativeWall.shape[-1]))
            #cond.shape    (batchsz=128) : (src_dim=maxObj=12)  : (dst_dim=maxObj=12)
            relativeZero = torch.zeros_like(relativeWall)
            relativeWall[cond] = relativeZero[cond]
        else:
            relativeWall = None

        if self.process_windoor:
            relativeWindoor = self.windoor(absoluteTensor[:,:,:self.bbox_dim], windoorTensor)
            cond = con.repeat((1, 1, relativeWindoor.shape[-1]))
            #cond.shape    (batchsz=128) : (src_dim=maxObj=12)  : (dst_dim=maxObj=12)
            relativeZero = torch.zeros_like(relativeWindoor)
            relativeWindoor[cond] = relativeZero[cond]
        else:
            relativeWindoor = None

        relativeObject = self.distance(absoluteTensor, True)
        con1 = con.reshape((self.batchsz, -1, 1, 1))
        con2 = con.reshape((self.batchsz,  1,-1, 1))
        cons = torch.logical_or(con1,con2)
        cond = cons.repeat((1, 1, 1, relativeObject.shape[-1]))
        relativeZero = torch.zeros_like(relativeObject)
        relativeObject[cond] = relativeZero[cond]

        return relativeObject, relativeWall, relativeWindoor

    def recon(self, absoluteTensor, denoise_out):
        #denoise_out is the difference in the objects' own co-ordinates of ABSOLUTETENSOR
        #
        abso_trans = absoluteTensor[:,:,0:self.translation_dim]
        abso_size = absoluteTensor[:,:,self.translation_dim:self.translation_dim+self.size_dim]
        abso_angle = absoluteTensor[:,:,self.bbox_dim-self.angle_dim:self.bbox_dim]
        abso_other = absoluteTensor[:,:,self.bbox_dim:]
        rela_trans = denoise_out[:,:,0:self.translation_dim]
        rela_size = denoise_out[:,:,self.translation_dim:self.translation_dim+self.size_dim]
        rela_angle = denoise_out[:,:,self.bbox_dim-self.angle_dim:self.bbox_dim]
        rela_other = denoise_out[:,:,self.bbox_dim:]
        new_trans = 0
        new_size = 0
        new_angle = 0
        new_other = abso_other + rela_other

        #scale
        if self.relativeScale:
            if self.relativeScaleMethod == "minus":
                new_size = abso_size + rela_size #add the x_recon's scale with denoise_out's scale
            if self.relativeScaleMethod == "divide":
                new_size = abso_size * rela_size #mutiply the x_recon's scale with denoise_out's scale
        else:
            new_size = rela_size #replace the x_recon's scale with denoise_out's scale
            
        if self.relativeScl_Ori_Trn:
            new_trans = abso_size * rela_trans #multiplies the denoise_out's translation with absoluteTensor's scale
        else:
            new_trans = rela_trans

        #rotate
        new_angle = F.normalize(self.simple_Norm_Minus(abso_angle, torch.cat([rela_angle[:,:,:1],-rela_angle[:,:,1:]],axis=-1)), dim=-1)
        
        new_trans = new_trans.reshape((self.batchsz, self.maxObj, 1, self.translation_dim))
        cs = abso_angle[:,:,:1].reshape((self.batchsz,-1,1,1))
        sn = abso_angle[:,:,1:].reshape((self.batchsz,-1,1,1))
        rotateX = torch.cat([cs, torch.zeros(cs.shape, device=cs.device), sn], axis = -1)
        #rotateX.shape    (batchsz=128) : (maxObj=12) : (spatial_dim_hori=1) : (spatial_dim_vert=3)  #vertical vector in spatial space
        rotateY = torch.cat([torch.zeros(cs.shape, device=cs.device), torch.ones(cs.shape, device=cs.device), torch.zeros(cs.shape, device=cs.device)], axis = -1)
        rotateZ = torch.cat([-sn, torch.zeros(cs.shape, device=cs.device), cs], axis = -1)
        new_trans = (torch.cat([rotateX, rotateY, rotateZ], axis = -2) * new_trans).sum(axis=-1)
        #rotate.shape    (batchsz=128) : (src_dim=maxObj=12) : (spatial_dim_hori=3) : (spatial_dim_vert=3)  #3*3 matrix in spatial space
        #new_trans.shape (batchsz=128) : (src_dim=maxObj=12) : (spatial_dim_hori=3) : (spatial_dim_vert=1)  #horizontal vector in spatial space
        new_trans = new_trans.reshape((self.batchsz,self.maxObj,3))

        #translate
        new_trans = abso_trans + new_trans

        return torch.cat([new_trans, new_size, new_angle, new_other], axis=-1)
