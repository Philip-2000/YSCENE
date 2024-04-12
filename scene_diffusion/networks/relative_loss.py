import numpy as np
import torch
import os
from scene_diffusion.networks.modules_yl import relaYL

class A():
    def get(self,nm,va):
        return va

check_flag = True
def logging(a):
    #if not check_flag:
    return
    for i in range(a.shape[0]):
        st = ""
        for j in range(a.shape[1]):
            st += "%.5f "%(a[i][j])
        print(st)

def preprocess_mat(data_start, mat_dis): #在这个函数里暂时还没有padding所以尺寸还不一样
    rela_calculator = relaYL(A())
    rela_calculator.batchsz=1
    rela_calculator.maxObj=len(data_start)
    rela_calculator.class_dim=1
    rela_calculator.point_dim=data_start.shape[-1]
    data_start = data_start.reshape((1,rela_calculator.maxObj,-1))
    mat = rela_calculator.distance(data_start, True)
    logging(mat_dis**2)

    ds1 = data_start.reshape((rela_calculator.maxObj,1,-1))
    ds2 = data_start.reshape((1,rela_calculator.maxObj,-1))

    dismat = ((((ds1-ds2)**2)[:,:,:rela_calculator.translation_dim]).sum(axis=-1))
    #dismat = ((((ds1-ds2)**2)[:,:,:1] + ((ds1-ds2)**2)[:,:,1:2] + ((ds1-ds2)**2)[:,:,2:3]))
    logging(dismat)

    matDis = (mat[:,:,:,:rela_calculator.translation_dim]**2).sum(axis=-1)[0]
    logging(matDis)
    if check_flag and ((matDis-mat_dis)**2).sum()>0.01:
        print("error")
    if check_flag and ((dismat-mat_dis)**2).sum()>0.01:
        print("fuck??")
    if check_flag and ((dismat-matDis)**2).sum()>0.001:
        print("error")
        return False, mat, dismat
    return True, mat, dismat

def relative_loss(relaCal, data_start, mat, mat_dis, data_recon):#这个函数里的mat和mat_dis都是一整个batch长度的，而且长度都做了padding
    mat_recon = relaCal.distance(data_recon, True)

    end_label = data_start[:,:,relaCal.bbox_dim+relaCal.class_dim-1:relaCal.bbox_dim+relaCal.class_dim] #30???????
    #end_label.shape : (batchsz = 128) : (ambiguous_dim = maxObj = 12) : (label_dim = 1)
    con = (end_label > torch.zeros_like(end_label))
    con1 = con.reshape((relaCal.batchsz,1,relaCal.maxObj)).repeat((1,relaCal.maxObj,1))
    con2 = con.reshape((relaCal.batchsz,relaCal.maxObj,1)).repeat((1,1,relaCal.maxObj))
    cons = torch.logical_and(con1,con2)
    cond = cons.reshape((relaCal.batchsz,relaCal.maxObj,relaCal.maxObj,1)).repeat((1,1,1,mat_recon.shape[-1]))
    #cond.shape    (batchsz=128) : (src_dim=1) : (dst_dim=maxObj=12) 
    
    #按照cond把mat_recon和mat相应的位置为0

    mat_reconZero = torch.zeros_like(mat_recon)
    mat_recon[cond] = mat_reconZero[cond]

    matZero = torch.zeros_like(mat)
    mat[cond] = matZero[cond]
    #求mat_recon和mat逐位求差。

    #按照mat_dis的倒数作为权重进行加权求和
    mat_d = torch.ones_like(mat_dis) / (torch.ones_like(mat_dis) + mat_dis)

    loss = ((mat_recon-mat)**2) * mat_d.reshape((relaCal.batchsz,relaCal.maxObj,relaCal.maxObj,1)).repeat((1,1,1,mat_recon.shape[-1]))

    return loss

a = "bed"
dsDir = "../data/3d_front_processed/"+a+"rooms_objfeats_32_64/"

def prepare_data_start(boxes):#(translation_dim=3)+(size_dim=3)+(angle_dim=2)
    return torch.Tensor(np.concatenate([boxes["translations"],boxes["sizes"],np.cos(boxes["angles"]),np.sin(boxes["angles"]),-np.ones_like(boxes["angles"])], axis=-1))
    
if __name__ == "__main__":

    cnt = 0
    for room in os.listdir(dsDir):
        #print(room)
        if room == "dataset_stats.txt":
            continue

        mat_dis = np.load(dsDir + room + "/matrix.npz", allow_pickle=True)["matrix"]
        boxes = np.load(dsDir + room + "/boxes.npz", allow_pickle=True)
        #if(mat_dis.shape[0]>4):
            #continue
        f, res, dismat = preprocess_mat(prepare_data_start(boxes), mat_dis)
        if not f:
            print(room)
            #break
        #np.savez_compressed(dsDir + room + "/matrix.npz", matrix=torch.tensor(dismat))
        #np.savez_compressed(dsDir + room + "/matrix_full.npz", matrix_full=torch.tensor(res))

        if cnt %100 == 99:
            print(cnt+1)
        cnt+=1


    #print(lens)
    #print("perfect = {}, unperfect = {}".format(perfect, unperfect))
