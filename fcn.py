from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,UpSampling2D
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.optimizers import SGD,Adadelta
import numpy as np
import matplotlib.pyplot as plt
from utils import load_data
from sklearn.model_selection import train_test_split

# build the model
model = Sequential()

# VGG like
N = 2
# block 1
model.add(Conv2D(4*N, (3, 3), activation='relu', padding='same', input_shape=(400, 400, 3)))
model.add(Conv2D(4*N, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# block 2
model.add(Conv2D(8*N, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(8*N, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# block 3
model.add(Conv2D(16*N, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16*N, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16*N, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# block 4
model.add(Conv2D(32*N, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32*N, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32*N, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# block 5
model.add(Conv2D(32*N, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32*N, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32*N, (3, 3), activation='relu', padding='same'))

# convolutional layers transferred from fully connected layer
model.add(Conv2D(256*N, (7, 7), activation='relu', padding='same',dilation_rate=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(256*N, (1, 1), activation='relu', padding='same'))
model.add(Dropout(0.5))

# classification layer
model.add(Conv2D(1, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1,1)))
model.add(UpSampling2D(size = (16,16)))
model.add(Activation('sigmoid'))

optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# optimizer = Nadam(lr=lr_base, beta_1 = 0.825, beta_2 = 0.99685)

loss_fn = ['mse']
metrics = ['accuracy']

model.compile(loss=loss_fn,
              optimizer=optimizer,
              metrics=metrics)

img_list,heatmap_list = load_data(N=2)
x_train,x_valid,y_train,y_valid = train_test_split(img_list,heatmap_list,test_size=0.3)

model.fit(x_train,y_train,epochs=2,batch_size=10)


# plt.subplot(121)
# plt.imshow(np.squeeze(model.predict(x_train[:1]),axis=(0,3)))
# plt.subplot(122)
# plt.imshow(np.squeeze(y_train[:1],axis=(0,3)))
#
from mask_to_submission import get_patch_label_from_array
# # train acc
# train_acc = []
# valid_acc = []
# for i in range(len(x_train)):
#     pre = np.squeeze(model.predict(x_train[i:i+1]), axis=(0,3))
#     train_patch_pre_label = np.array(list(get_patch_label_from_array(pre,th=0.21)))
#     ground_truth_label = np.array(list(get_patch_label_from_array(y_train[i,:,:,0],th=0.21)))
#     train_acc.append(np.mean(np.equal(train_patch_pre_label[:,2],ground_truth_label[:,2])))
#
# # valid acc
# for i in range(len(x_valid)):
#     pre = np.squeeze(model.predict(x_valid[i:i+1]), axis=(0,3))
#     train_patch_pre_label = np.array(list(get_patch_label_from_array(pre,th=0.21)))
#     ground_truth_label = np.array(list(get_patch_label_from_array(y_valid[i,:,:,0],th=0.21)))
#     train_acc.append(np.mean(np.equal(train_patch_pre_label[:,2],ground_truth_label[:,2])))

# make submission
from utils import load_test_image
import cv2 as cv
test_mat,test_img_id = load_test_image()

th = 0.21
f = open('submission.csv','w')
f.write('id,prediction\n')
for i in range(len(test_img_id)):
    print(i)
    img = np.expand_dims(cv.resize(test_mat[i], dsize=(400, 400), interpolation=cv.INTER_CUBIC),axis=0)
    pre = np.squeeze(model.predict(img),axis=(0,3))
    pre_resize = cv.resize(pre,dsize=(test_mat[0].shape[0],test_mat[0].shape[1]),interpolation=cv.INTER_CUBIC)
    train_patch_pre_label = list(get_patch_label_from_array(pre_resize,th=th))
    for line in train_patch_pre_label:
        final_string = "{:03d}_{}_{},{}".format(test_img_id[i], line[0], line[1], line[2])
        f.write(final_string+'\n')
