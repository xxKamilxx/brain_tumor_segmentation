
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from skimage.transform import resize
from keras.models import Model, load_model
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import cv2
import os
import natsort
src="C:/Users/kamil/Desktop/MLnauka/brain tumor dataset"

img_list=os.listdir(src)
srt_list=natsort.natsorted(img_list)
images,masks=[],[]
word="mask" #sentence to find in path

for file in srt_list:

    if word in file:
        mask = cv2.imread(os.path.join(src,file),0)
        mask = resize(mask, (128, 128, 1),mode = 'constant', preserve_range = True)
        mask = np.asarray(mask)
        masks.append(mask)


    else:
        print(file)
        img = cv2.imread(os.path.join(src,file))
        img= resize(img, (128, 128, 3),mode = 'constant', preserve_range = True)
        img= np.asarray(img)
        images.append(img)


images_array=np.array(images)/255.0
masks_array=np.array(masks)/255.0

x_train,y_train, x_valid,y_valid=train_test_split(images_array,masks_array,test_size=0.2,random_state=1)

pickle.dump(x_train,open('x_train.pkl','wb'))
pickle.dump(y_train,open('y_train.pkl','wb'))
pickle.dump(x_valid,open('x_valid.pkl','wb'))
pickle.dump(y_valid,open('y_valid.pkl','wb'))