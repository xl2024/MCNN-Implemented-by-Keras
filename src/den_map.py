import numpy as np
import cv2
import random
from scipy.ndimage import filters


def data_norm(dataset,labels):
    _dataset=[]
    _labels=[]
    for i in range(len(dataset)):
        pic=np.float32(dataset[i])
        if np.max(pic)!=np.min(pic) and np.std(pic)!=0:
            pic     = (pic-np.min(pic))/(np.max(pic)-np.min(pic))
            pic     = (pic-np.mean(pic))/np.std(pic)
            _dataset.append(pic)
            _labels.append(labels[i])
    h,w,c=pic.shape[0],pic.shape[1],pic.shape[2]
    _dataset=(np.array(_dataset)).reshape((-1,h,w,c))
    h,w=labels.shape[1:3]
    _labels=(np.array(_labels)).reshape((-1,h,w,1))
    return _dataset,_labels
    

def rflip(img1,img2):
    ranflip=random.randint(0,3)
    if ranflip==0:
        img1=cv2.flip(img1,0)
        img2=cv2.flip(img2,0)
    if ranflip==1:
        img1=cv2.flip(img1,1)
        img2=cv2.flip(img2,1)
    if ranflip==2:
        img1=cv2.flip(img1,-1)
        img2=cv2.flip(img2,-1)
    img2=img2.reshape((img2.shape[0],img2.shape[1],1))
    return img1,img2


def shuffle_data(dataset,labels):
    shuffle = [i for i in range(len(dataset))] 
    shuffle = random.sample(shuffle, len(shuffle))
    t1      = dataset[shuffle[0]].copy()
    t2      = labels[shuffle[0]].copy()
    for i in range(len(shuffle)-1):
        dataset[shuffle[i]] = dataset[shuffle[i+1]].copy()
        labels[shuffle[i]]  = labels[shuffle[i+1]].copy()
        dataset[shuffle[i]],labels[shuffle[i]]=rflip(dataset[shuffle[i]],labels[shuffle[i]])
    t1,t2=rflip(t1,t2)
    dataset[shuffle[len(shuffle)-1]] = t1
    labels[shuffle[len(shuffle)-1]]  = t2
    return dataset, labels


def den(img_shape=None,gt=None,width=320,height=180):        #produce the density map
    beta=0.3
    dist=np.array([])
    w,h,c=int(width/4+0.5),int(height/4+0.5),img_shape[2]
    den_map=np.zeros((h,w),dtype=np.float64)
    n=gt.shape[0]                 #the number of the labeled heads
    scal=width/img_shape[1]
    for i in range(n):
        for j in range(n):        #find the two nearest heads from the i-th head
            if i != j:
                dist=np.append(dist,np.sqrt(np.sum(np.square(gt[i]-gt[j]))))
        dist.sort()
        dist=dist[0:2]            #set m=2
        dist=np.mean(dist)*scal/4
        x,y=gt[i]*scal/4
        x,y=int(x+0.5),int(y+0.5)
        if x==h:
            x=x-1
        if y==w:
            y=y-1
        k = int(beta*dist+0.5)
        if k==0:
            k=1
        dk=int(k/2)
        while x-dk<0 or x+dk>=h or y-dk<0 or y+dk>=w:
            dk-=1
        k=dk*2+1
        kn=np.zeros((dk*2+1,dk*2+1),dtype=np.float64)
        kn[dk][dk]=1.
        den_map[x-dk:x+dk+1,y-dk:y+dk+1]+=filters.gaussian_filter(kn,k)
    return den_map,n


if __name__=='__main__':
    img=cv2.imread('./pics1/154save_name.jpg')
    txt=open('./labels1/labels_154.txt','r')
    label=np.array([])
    for line in txt:
        label=np.append(label,eval(line))
    txt.close()
    label=label.reshape((-1,2))
    den_map,n=den(img.shape,label)
    print(np.sum(den_map))
    cv2.imwrite(str(n)+'_154result.jpg',den_map*255)
