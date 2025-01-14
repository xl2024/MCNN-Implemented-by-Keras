from keras.models import Model, Sequential, load_model
from keras.layers import Input, Activation
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import os
import datetime
from keras.optimizers import Adam,SGD
from keras.layers import Concatenate

import cv2
from math import ceil

from den_map import *


train='pics0/'
gt='labels0/'
val='val_pics0/'
val_gt='labels0/'
pretrain='pretrain.h5'
adam=Adam(lr=0.00001)
optimizer=adam
epochs=200
width=256
height=144

print(datetime.datetime.now(),'loading')

def get_labels(count=0,train=None,gt=None,width=0,height=0):
    dataset=np.array([])
    counts=np.array([],dtype=int)
    labels=np.array([])
    for lis in os.listdir(train):
        count=count+1
        if count % 10 == 0 :
            print(datetime.datetime.now(),'processing the '+str(count)+'th image...')
        img=cv2.imread(os.path.join(train, lis))
        label=np.array([],dtype=int)
        txt=open(os.path.join(gt,'{}{}'.format(lis[0:-13],'save_name.txt')),'r')
        #txt=open(os.path.join(gt,'labels_'+lis[0:3]+'.txt'),'r')
        for line in txt:
            label=np.append(label,eval(line))
        txt.close()
        label=label.reshape((-1,2))
        den_map,n=den(img.shape,label,width,height)
        labels=np.append(labels, den_map)
        img=cv2.resize(img, (width,height))
        cv2.imwrite('forlis.jpg',img)
        dataset=np.append(dataset, img)
        counts=np.append(counts,n)
    channel=img.shape[2]
    dataset=dataset.reshape((-1,height,width,channel))
    labels=labels.reshape((-1,int(height/4+0.5),int(width/4+0.5),1))
    dataset,labels=data_norm(dataset,labels)
    dataset,labels=shuffle_data(dataset,labels)
    cv2.imwrite('last_data.jpg',(dataset[-1]+1)*127.5)
    cv2.imwrite('last_label.jpg',labels[-1]*255)
    return dataset,counts,labels,channel

dataset,counts,labels,channel=get_labels(train=train,
                                             gt=gt,
                                             width=width,
                                             height=height)

print(datetime.datetime.now(),'training')

inputs=Input(shape=(height,width,channel))

x1=Conv2D(16,(9,9),padding='same',activation='relu',name='11')(inputs)
x1=MaxPooling2D((2,2),padding='same')(x1)
x1=Conv2D(32,(7,7),padding='same',activation='relu',name='12')(x1)
x1=MaxPooling2D((2,2),padding='same')(x1)
x1=Conv2D(16,(7,7),padding='same',activation='relu',name='13')(x1)
x1=Conv2D(8,(7,7),padding='same',activation='relu',name='14')(x1)

x2=Conv2D(20,(7,7),padding='same',activation='relu',name='21')(inputs)
x2=MaxPooling2D((2,2),padding='same')(x2)
x2=Conv2D(40,(5,5),padding='same',activation='relu',name='22')(x2)
x2=MaxPooling2D((2,2),padding='same')(x2)
x2=Conv2D(20,(5,5),padding='same',activation='relu',name='23')(x2)
x2=Conv2D(10,(5,5),padding='same',activation='relu',name='24')(x2)

x3=Conv2D(24,(5,5),padding='same',activation='relu',name='31')(inputs)
x3=MaxPooling2D((2,2),padding='same')(x3)
x3=Conv2D(48,(3,3),padding='same',activation='relu',name='32')(x3)
x3=MaxPooling2D((2,2),padding='same')(x3)
x3=Conv2D(24,(3,3),padding='same',activation='relu',name='33')(x3)
x3=Conv2D(12,(3,3),padding='same',activation='relu',name='34')(x3)

x4=Conv2D(12,(11,11),padding='same',activation='relu',name='41')(inputs)
x4=MaxPooling2D((2,2),padding='same')(x4)
x4=Conv2D(24,(9,9),padding='same',activation='relu',name='42')(x4)
x4=MaxPooling2D((2,2),padding='same')(x4)
x4=Conv2D(12,(9,9),padding='same',activation='relu',name='43')(x4)
x4=Conv2D(6,(9,9),padding='same',activation='relu',name='44')(x4)

x5=Conv2D(8,(13,13),padding='same',activation='relu',name='51')(inputs)
x5=MaxPooling2D((2,2),padding='same')(x5)
x5=Conv2D(16,(11,11),padding='same',activation='relu',name='52')(x5)
x5=MaxPooling2D((2,2),padding='same')(x5)
x5=Conv2D(8,(11,11),padding='same',activation='relu',name='53')(x5)
x5=Conv2D(4,(11,11),padding='same',activation='relu',name='54')(x5)

x=Concatenate()([x3,x2,x1,x4,x5])

column1=Sequential()
column1=load_model('x1_97.h5')
model1=Model(input=column1.input,output=column1.get_layer('14').output)
column2=Sequential()
column2=load_model('x2_75.h5')
model2=Model(input=column2.input,output=column2.get_layer('24').output)
column3=Sequential()
column3=load_model('x3_53.h5')
model3=Model(input=column3.input,output=column3.get_layer('34').output)
x1=model1(inputs)
x2=model2(inputs)
x3=model3(inputs)
x=Concatenate()([x1,x2,x3])
predictions=Conv2D(1,(1,1),padding='same',name='4')(x)

model=Model(inputs=inputs,outputs=predictions)

model.compile(loss='mse',
              optimizer=optimizer,
              metrics=['mae','mse'])
model.summary()
model.fit(dataset, labels, epochs=epochs, batch_size=32)
pred=model.predict(dataset, batch_size=32)
num_pred=np.array([],dtype=int)
num_labels=np.array([],dtype=int)
for i in labels:
    num_labels=np.append(num_labels,int(np.sum(i)))
for i in pred:
    num_pred=np.append(num_pred,int(np.sum(i)))
cv2.imwrite('pred.jpg',i*255)


def save_txt(name,content):
    txt=open(name,'w')
    content='\n'.join(str(i) for i in content)
    txt.write(content)
    txt.close()
  
save_txt('num_counts.txt',counts)
save_txt('num_labels.txt',num_labels)
save_txt('num_predictions.txt',num_pred)
#save_txt('labels.txt',labels)
#save_txt('pred.txt',pred)
print('predictions =',num_pred)

model.save(pretrain)

print(datetime.datetime.now(),'testing')

val_dataset,val_counts,val_labels,channel=get_labels(train=val,
                                                         gt=val_gt,
                                                        width=width,
                                                        height=height)
score=model.evaluate(val_dataset,val_labels,batch_size=32)
print(model.metrics_names,'=',score)   

print(datetime.datetime.now(),'the end')
