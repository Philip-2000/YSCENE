import torch
import torch.nn.functional as F


class classifier_guidance():
    def __init__(self, config, betas):
        #all kinds of dimensions here
        self.batchsz = config.get("batchsz", 4)

        #................basic length of the data structure...................
            #.........absoluteTensor.....................
        self.translation_dim = config.get("translation_dim", 3)
        self.size_dim = config.get("size_dim", 3)
        self.angle_dim = config.get("angle_dim", 2)
        self.bbox_dim = self.translation_dim + self.size_dim + self.angle_dim
        self.maxObj = config.get("sample_num_points", 21)

            #.........wallTensor.....................
        self.process_wall = config.get("process_wall", False)
        self.wall_translation_dim = config.get("wall_translation_dim", 2)
        self.wall_norm_dim = config.get("wall_norm_dim",2)
        self.wall_dim = self.wall_translation_dim + self.wall_norm_dim
        self.maxWall = config.get("maxWall", 16)

        #^^^^^^^^^^^^^^^^^^^^^basic length of the data structure^^^^^^^^^^^^^^^^^^^

        if self.angle_dim != 2 or self.wall_norm_dim != 2:
            raise NotImplementedError

        self.temp = torch.Tensor([[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]])
        self.sample_dim = self.temp.shape[0]
        self.border = 0.1
        self.maxT = 0.5
        pass
    
    def flatten(self, absolute):
        #batchsz = 128 : maxObj = 12 : bbox_dim = 8
        #

        tras = torch.cat([absolute[:,:,0:1],absolute[:,:,self.translation_dim-1:self.translation_dim]], axis=-1).reshape((self.batchsz,self.maxObj,1,2))
        sizs = torch.cat([absolute[:,:,self.translation_dim:self.translation_dim+1],absolute[:,:,self.translation_dim+self.size_dim-1:self.translation_dim+self.size_dim]], axis=-1).reshape((self.batchsz,self.maxObj,1,1,2))
        coss = absolute[:,:,self.bbox_dim-2:self.bbox_dim-1]
        sins = absolute[:,:,self.bbox_dim-1:self.bbox_dim]
        mat1 = torch.cat([coss,sins], axis=-1).reshape((self.batchsz,self.maxObj,1,1,2))
        mat2 = torch.cat([-sins,coss], axis=-1).reshape((self.batchsz,self.maxObj,1,1,2))
        mats = torch.cat([mat1,mat2], axis=-2).reshape((self.batchsz,self.maxObj,1,2,2))
        templa = self.temp.reshape((1,1,-1,1,2))

        directions = (sizs * templa * mats).sum(axis=-1).reshape((self.batchsz,self.maxObj,-1,2))
        locations = directions + tras
        #batchsz = 128 : maxObj = 12 : sample_dim = 8 : location_dim = 2
        return directions, locations

    def squeeze(self, wallTensor):
        return wallTensor
        #wall: batchsz = 128 : maxWall = 16 : location_dim+norm_dim = 2 + 2 = 4
        wallPoint = wallTensor[:,:,:2]
        wallNorm = torch.cat([wallTensor[:,:,3:4],wallTensor[:,:,2:3]],axis=-1)
        wallPreNo = torch.cat([wallNorm[:,-1:,:],wallNorm[:,:-1,:]], axis=-2)
        wallPoint += self.border * wallNorm + self.border * wallPreNo 

        return torch.cat([wallPoint,wallTensor[:,:,2:]],axis=-1)

    def field(self, locations, wallTensor):
        wall = self.squeeze(wallTensor)
        #wall: batchsz = 128 : maxWall = 16 : location_dim+norm_dim = 2 + 2 = 4
        
        #locations: batchsz = 128 : maxObj = 12 : sample_dim = 8 : location_dim = 2

        flatten_location = locations.reshape((self.batchsz, -1, 1, 2))
        flatten_wall_loc = wall[:,:,:2].reshape((self.batchsz, 1, -1, 2))
        flatten_wall_dir = torch.cat([wall[:,:,3:4],wall[:,:,2:3]],axis=-1).reshape((self.batchsz, 1, -1, 2))
        loca = flatten_wall_loc - flatten_location
        locb = torch.cat([flatten_wall_loc[:,:,1:,:],flatten_wall_loc[:,:,:1,:]], axis=-2) - flatten_location
        relative_inn = (loca*locb).sum(axis=-1)
        relative_len = ((loca**2).sum(axis=-1) * (locb**2).sum(axis=-1))**0.5
        relative_ori = torch.arccos(relative_inn / relative_len).reshape((self.batchsz, -1, self.maxWall))
        relative_cro = torch.sign(loca[:,:,:,0:1] * locb[:,:,:,1:2] - loca[:,:,:,1:2] * locb[:,:,:,0:1]).reshape((self.batchsz, -1, self.maxWall))
        
        circle_ori = (relative_cro * relative_ori).sum(axis=-1)
        inRoom = (torch.abs(circle_ori) > torch.ones_like(circle_ori) * 0.001)
        inRoom = inRoom.reshape((self.batchsz, -1, 1)).repeat((1,1,2))
        #inRoom: batchsz = 128 : maxObj*sample_dim = 96 : inRoom = 1

        to_wall_dir_proj = (flatten_wall_dir * loca).sum(axis=-1)
        inWall = to_wall_dir_proj < torch.zeros_like(to_wall_dir_proj)
        #inWall: batchsz = 128 : maxObj*sample_dim = 96 : maxWall = 16 : inWall = 1

        locc = to_wall_dir_proj.reshape((self.batchsz, -1, self.maxWall, 1)) * flatten_wall_dir
        inValidWall = ((loca - locb)**2).sum(axis=-1) < 0.001 * torch.ones_like(inWall)
        zeroMoveWeight = torch.logical_or(inWall, inValidWall)
        #print(zeroMoveWeight)

        sidePointDir = ((loca-locc)*(locb-locc)).sum(axis=-1)

        insideWall = (sidePointDir < torch.zeros_like(sidePointDir))
        #print(insideWall)
        insideWall = insideWall.reshape((self.batchsz, -1, self.maxWall, 1)).repeat((1,1,1,2))
        #insideWall: batchsz = 128 : maxObj*sample_dim = 96 : maxWall = 16 : insideWall = 1

        lena = (loca**2).sum(axis=-1)
        lenb = (locb**2).sum(axis=-1)
        
        mina = (lena < lenb).reshape((self.batchsz, -1, self.maxWall, 1)).repeat((1,1,1,2))
        pointWallMove = locb
        pointWallMove[mina] = loca[mina]
        pointWallMove[insideWall] = locc[insideWall]
        #pointWallMove: batchsz = 128 : maxObj*sample_dim = 96 : maxWall = 16 : location_dim = 2

        # print(inWall.shape)
        pointWallMoveWeight = torch.exp(-(pointWallMove**2).sum(axis=-1))
        pointWallMoveWeight[zeroMoveWeight] = torch.zeros_like(pointWallMoveWeight)[zeroMoveWeight]
        pointWallMoveWeight = (F.normalize(pointWallMoveWeight, dim=-1)**2).reshape((self.batchsz, -1, self.maxWall, 1)).repeat((1,1,1,2))
        #pointWallMove: batchsz = 128 : maxObj*sample_dim = 96 : maxWall = 16 : exp(-len) = 1

        pointMove = (pointWallMove * pointWallMoveWeight).sum(axis=-2)
        #pointMove: batchsz = 128 : maxObj*sample_dim = 96 : location_dim = 2
        pointMove[inRoom] = torch.zeros_like(pointMove)[inRoom]

        return pointMove.reshape((self.batchsz, -1, self.sample_dim, 2))

    def synthesis(self, directions, fields, absolute, t):
        #absolute: batchsz = 128 : maxObj = 12 : bbox_dim = 8
        
        #directions / locations / fields: batchsz = 128 : maxObj = 12 : sample_dim = 8 : location_dim = 2

        #综合各个点情况，形成box的。先这样写吧。以后可以调。
            #translate
                #各个点的位移的平均值？
        translate = fields.mean(axis=-2)
        #directions / locations / fields: batchsz = 128 : maxObj = 12 : location_dim = 2
        # print("translate.shape")
        # print(translate.shape)
            #rotate
                #各个点的位移的旋度？
        normals = F.normalize(torch.cat([-directions[:,:,:,1:], directions[:,:,:,:1]],axis=-1),dim=-1)
        
        # print("normals.shape")
        # print(normals.shape)
        rotate = ((normals * fields).sum(axis=-1) / (directions**2).sum(axis=-1)**0.5).mean(axis=-1)
        # print("rotate.shape")
        # print(rotate.shape)
        #directions / locations / fields: batchsz = 128 : maxObj = 12
            

            #scale
            #scale: batchsz = 128 : maxObj = 12 : scale_dim = 2
                #各个点的位移的散度？
        absoluteVector = absolute[:,:,self.bbox_dim-self.angle_dim:self.bbox_dim].reshape((self.batchsz, self.maxObj, 1, self.angle_dim))
        coss = torch.cos(rotate * t * 0.5).reshape((self.batchsz,-1,1,1))
        sins = torch.sin(rotate * t * 0.5).reshape((self.batchsz,-1,1,1))
        mat1 = torch.cat([coss,sins], axis=-1).reshape((self.batchsz,self.maxObj,1,2))
        mat2 = torch.cat([-sins,coss], axis=-1).reshape((self.batchsz,self.maxObj,1,2))
        mats = torch.cat([mat1,mat2], axis=-2).reshape((self.batchsz,self.maxObj,2,2))
        absoluteVector = (mats * absoluteVector).sum(axis=-1).reshape((self.batchsz, self.maxObj, self.angle_dim))
        
        mat1 = torch.cat([absoluteVector[:,:,:1],-absoluteVector[:,:,1:]], axis=-1).reshape((self.batchsz,self.maxObj,1,2))
        mat2 = torch.cat([absoluteVector[:,:,1:],absoluteVector[:,:,:1]], axis=-1).reshape((self.batchsz,self.maxObj,1,2))
        mats = torch.cat([mat1,mat2], axis=-2).reshape((self.batchsz,self.maxObj,1,2,2))#.repeat(1,1,self.sample_dim,1,1)
        #print(fields[2][4])
        rotatedFields = (mats * fields.reshape((self.batchsz,self.maxObj,-1,1,2))).sum(axis=-1).reshape((self.batchsz, self.maxObj, -1, 2))
        #print(rotatedFields[2][4])
        originalDirection = F.normalize(self.temp,dim=-1).reshape((1,1,-1,2))
        
        scale = (originalDirection * rotatedFields).mean(axis=-2) * (4/3)
        #print(scale[2][4])
        
        absolute[:,:,0:1] += translate[:,:,0:1] * t.reshape((-1,1,1))
        absolute[:,:,self.translation_dim-1:self.translation_dim] += translate[:,:,1:2] * t.reshape((-1,1,1))
        absolute[:,:,self.translation_dim:self.translation_dim+1] += scale[:,:,0:1] * t.reshape((-1,1,1))
        absolute[:,:,self.translation_dim+self.size_dim-1:self.translation_dim+self.size_dim] += scale[:,:,1:2] * t.reshape((-1,1,1))

        #absoluteVector: batchsz = 128 : maxObj = 12 : angle_dim = 2
        absoluteVector = absolute[:,:,self.bbox_dim-self.angle_dim:self.bbox_dim].reshape((self.batchsz, self.maxObj, 1, self.angle_dim))
        coss = torch.cos(rotate * t).reshape((self.batchsz,-1,1,1))
        sins = torch.sin(rotate * t).reshape((self.batchsz,-1,1,1))
        mat1 = torch.cat([coss,sins], axis=-1).reshape((self.batchsz,self.maxObj,1,2))
        mat2 = torch.cat([-sins,coss], axis=-1).reshape((self.batchsz,self.maxObj,1,2))
        mats = torch.cat([mat1,mat2], axis=-2).reshape((self.batchsz,self.maxObj,2,2))
        absoluteVector = (mats * absoluteVector).sum(axis=-1).reshape((self.batchsz, self.maxObj, self.angle_dim))
        absolute[:,:,self.bbox_dim-self.angle_dim:self.bbox_dim] = absoluteVector

        return absolute, translate, mats, scale
        #denoise_out is the difference in the objects' own co-ordinates of ABSOLUTETENSOR

    def tArrangement(self,t):
        return (self.maxT*(1. - t / torch.float(self.betas.shape[0]))).reshape((-1,1))


    def full(self, absolute, wallTensor, t):
        directions, locations = self.flatten(absolute)
        # print("locations.shape")
        # print(locations.shape)
        fields = self.field(locations, wallTensor)
        # print("fields.shape")
        # print(fields.shape)
        result, tr, ro, sc = self.synthesis(directions, fields, absolute, self.tArrangement(t))
        return result


