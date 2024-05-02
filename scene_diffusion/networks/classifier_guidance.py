
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



class classifier_guidance():
    def __init__(self, config):
        #all kinds of dimensions here
        self.batchsz = config.get("batchsz", 128)

        #................basic length of the data structure...................
            #.........absoluteTensor.....................
        self.objectness_dim = config.get("objectness_dim", 0) #exist or not, if objectness_dim=0, 
                                            #then we use the last digit of class_dim (or "end_label") as the objectness
        self.class_dim = config.get("class_dim", 25)
        self.use_weight = config.get("use_weight", False)
        self.weight_dim = config.get("weight_dim", 32)
        self.translation_dim = config.get("translation_dim", 3)
        self.size_dim = config.get("size_dim", 3)
        self.angle_dim = config.get("angle_dim", 2)
        self.bbox_dim = self.translation_dim + self.size_dim + self.angle_dim
        self.objfeat_dim = config.get("objfeat_dim", 0)
        self.point_dim = config.get("point_dim", 65)
        self.maxObj = config.get("sample_num_points", 21)
        if self.use_weight:
            self.class_dim = self.weight_dim
        
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
    
    def recon(self, absoluteTensor, denoise_out):
        return 
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
