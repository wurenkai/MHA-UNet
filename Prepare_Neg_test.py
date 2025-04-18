# -*- coding: utf-8 -*-
import h5py
import numpy as np
import scipy.io as sio
import scipy.misc as sc
import glob


# Parameters
height = 256
width  = 256
channels = 3

############################################################# Prepare NormalSkin data set #################################################


Tr_list = glob.glob("Normal(Healthy skin)"+'/*.png')
Data_train_2018    = np.zeros([200, height, width, channels])
Label_train_2018   = np.zeros([200, height, width])

print(len(Tr_list))
for idx in range(len(Tr_list)):
    print(idx+1)
    img = sc.imread(Tr_list[idx])
    img = np.double(sc.imresize(img, [height, width, channels], interp='bilinear', mode = 'RGB'))
    Data_train_2018[idx, :,:,:] = img

    b = Tr_list[idx]
    b = b[len(b)-8: len(b)-4]
    add = ("masks_black/" + b +'.png')
    img2 = sc.imread(add)
    img2 = np.double(sc.imresize(img2, [height, width], interp='bilinear'))
    Label_train_2018[idx, :,:] = img2    
         
print('finished')

Test_img       = Data_train_2018[0:200,:,:,:]

Test_mask       = Label_train_2018[0:200,:,:]


np.save('data_test' , Test_img)

np.save('mask_test' , Test_mask)


