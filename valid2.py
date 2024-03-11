import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import random
import os

import tensorflow.keras.backend as K
import tensorflow.keras.utils

from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Activation, Add, Conv2D, Conv2DTranspose, concatenate, Cropping2D, MaxPooling2D, Reshape, UpSampling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.regularizers import l2

class MaskGenerator(tensorflow.keras.utils.Sequence):

    def __init__(self, directory, idxs, img_dict, labels_dict,
                 target_size=(512,64), batch_size=1, augment=False,
                 transform_dict = None, shuffle=True):

        self.directory = directory
        self.idxs = idxs
        self.img_dict = img_dict
        self.labels_dict = labels_dict
        self.transform_dict = transform_dict
        self.target_size = target_size
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.on_epoch_end()

    # shuffle indices at the end of each epoch
    def on_epoch_end(self):
        if self.shuffle is True:
            np.random.shuffle(self.idxs)

    # return number of batches per epoch
    def __len__(self):

        if self.augment is True:
            multiplier = 5
        else:
            multiplier = 1

        return int(np.floor(len(self.idxs) * multiplier / self.batch_size))

    # check if transformed point is located within image boundaries
    def _checkBoundaries(self, p):

        # x dimension
        if p[0] < 0:
            px = 0
        elif p[0] > self.target_size[0]:
            px = self.target_size[0]
        else:
            px = p[0]

        # y dimension
        if p[1] < 0:
            py = 0
        elif p[1] > self.target_size[1]:
            py = self.target_size[1]
        else:
            py = p[1]

        return (int(px), int(py))

    # apply shifts, rotations, scaling and flips to original image and keypoints
    def _transform_image(self, img, keypoints):

        aug_keypoints = []

        c = (img.shape[0] // 2, img.shape[1] // 2)

        if self.transform_dict['Flip']:
            flip = random.choice([True, False])
            if flip:
                img = cv.flip(img, flipCode=1)

        if self.transform_dict['Rotate']:

            if self.transform_dict['Scale']:
                s = random.uniform(0.7, 1.0)
            else:
                s = 1.0

            r = random.randint(-10, 10)
            M_rot = cv.getRotationMatrix2D(center=(img.shape[0] // 2, img.shape[1] // 2), angle=r, scale=s)
            img = cv.warpAffine(img, M_rot, (img.shape[0], img.shape[1]), borderMode=cv.BORDER_CONSTANT, borderValue=1)

        if self.transform_dict['Shift']:
            tx = random.randint(-20, 20)
            ty = random.randint(-20, 20)
            M_shift = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
            img = cv.warpAffine(img, M_shift, (img.shape[0], img.shape[1]),
                                borderMode=cv.BORDER_CONSTANT, borderValue=1)

        # transform keypoints
        c = (img.shape[0] // 2, img.shape[1] // 2)

        for i in range(0, len(keypoints) - 1, 2):

            px = keypoints[i]
            py = keypoints[i+1]
            p = np.array([px, py, 1], dtype=int)

            # apply flip
            if self.transform_dict['Flip'] and flip:
                p[0] = c[0] - (p[0] - c[0])

            # apply rotation
            if self.transform_dict['Rotate']:
                p = np.dot(M_rot, p)

            # apply horizontal / vertical shifts
            if self.transform_dict['Shift']:
                p[0] += tx
                p[1] += ty

            p = self._checkBoundaries(p)

            aug_keypoints.append(p[0])
            aug_keypoints.append(p[1])

        return img, aug_keypoints

    # load image from disk
    def _load_image(self, fn):

        img = cv.imread(filename=os.path.join(self.directory, fn))
        # print(os.path.join(self.directory, fn), img)
        # print(fn)
        # img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = np.float32(img) / 255

        return img

    # apply gaussian kernel to image
    def _gaussian(self, xL, yL, sigma, H, W):

        channel = [math.exp(-((c - xL) ** 2 + (r - yL) ** 2) / (2 * sigma ** 2)) for r in range(H) for c in range(W)]
        channel = np.array(channel, dtype=np.float32)
        channel = np.reshape(channel, newshape=(H, W))

        return channel

    # convert original image to heatmap
    def _convertToHM(self, img, keypoints, sigma=5):

        H = img.shape[0]
        W = img.shape[1]
        nKeypoints = len(keypoints)

        img_hm = np.zeros(shape=(H, W, nKeypoints // 2), dtype=np.float32)

        for i in range(0, nKeypoints // 2):
            x = keypoints[i * 2]
            y = keypoints[1 + 2 * i]

            channel_hm = self._gaussian(x, y, sigma, H, W)

            img_hm[:, :, i] = channel_hm
        
        img_hm = np.reshape(img_hm, newshape=(img_hm.shape[0]*img_hm.shape[1]*nKeypoints // 2, 1))

        return img_hm

    # generate batches of scaled images and bounding boxes
    def _data_generation(self, idxs):

        x = []
        y = []

        for idx in idxs:
            img = self._load_image(self.img_dict[idx])
            keypoints = self.labels_dict[idx]

            if self.augment is True and self.transform_dict:
                img, keypoints = self._transform_image(img, keypoints)

            img = np.reshape(img, (512, 64, 1))
            img_hm = self._convertToHM(img, keypoints)

            x.append(img)
            y.append(img_hm)

        return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

    # return indices for train batches
    def _get_train_idxs(self, idx):

        # number of batches in original train set
        N = int(np.floor(len(self.idxs) / self.batch_size))

        # idx exceeds original image indices
        if idx > N:

            # reset start idx
            if idx % N == 0:
                reset_idx = 0 #((idx - 1) % N) + 1
            else:
                reset_idx = idx % N - 1

            start = reset_idx * self.batch_size

            # end idx
            if (reset_idx + 1) * self.batch_size > len(self.idxs):
                end = len(self.idxs)
            else:
                end = (reset_idx + 1) * self.batch_size

        # idx is within in original train set
        else:
            start = idx * self.batch_size
            end = (idx + 1) * self.batch_size

        return start, end

    # return indices for val batches
    def _get_val_idxs(self, idx):

        if (idx + 1) * self.batch_size > len(self.idxs):
            end = len(self.idxs)
        else:
            end = (idx + 1) * self.batch_size

        return idx * self.batch_size, end

    # return batch of image data and labels
    def __getitem__(self, idx):

        if self.augment is True:
            start_batch_idx, end_batch_idx = self._get_train_idxs(idx)
        else:
            start_batch_idx, end_batch_idx = self._get_val_idxs(idx)

        idxs = self.idxs[start_batch_idx:end_batch_idx]
        batch_x, batch_y = self._data_generation(idxs)

        return batch_x, batch_y

def UNET(input_shape):
    def downsample_block(x, block_num, n_filters, pooling_on=True):

        x = Conv2D(n_filters, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                   name="Block" + str(block_num) + "_Conv1")(x)
        x = Conv2D(n_filters, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                   name="Block" + str(block_num) + "_Conv2")(x)
        skip = x

        if pooling_on is True:
            x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', name="Block" + str(block_num) + "_Pool1")(x)

        return x, skip

    def upsample_block(x, skip, block_num, n_filters):

        x = Conv2DTranspose(n_filters, kernel_size=(2, 2), strides=2, padding='valid', activation='relu',
                            name="Block" + str(block_num) + "_ConvT1")(x)
        x = concatenate([x, skip], axis=-1, name="Block" + str(block_num) + "_Concat1")
        x = Conv2D(n_filters, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                   name="Block" + str(block_num) + "_Conv1")(x)
        x = Conv2D(n_filters, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                   name="Block" + str(block_num) + "_Conv2")(x)

        return x

    input = Input(input_shape, name="Input")

    # downsampling
    x, skip1 = downsample_block(input, 1, 64)
    x, skip2 = downsample_block(x, 2, 128)
    x, skip3 = downsample_block(x, 3, 256)
    x, skip4 = downsample_block(x, 4, 512)
    x, _ = downsample_block(x, 5, 512, pooling_on=False)
    # upsampling
    x = upsample_block(x, skip4, 6, 256)
    x = upsample_block(x, skip3, 7, 128)
    x = upsample_block(x, skip2, 8, 64)
    x = upsample_block(x, skip1, 9, 64)
    output = Conv2D(1, kernel_size=(1, 1), strides=1, padding='valid', activation='linear', name="output")(x)
    output = Reshape(target_shape=(H*W*Nkeypoints,1))(output)


    model = Model(inputs=input, outputs=output, name="Output")

    return model

def maskToKeypoints(mask):
    # mask = np.reshape(mask, newshape=(96,96))
    kp = np.unravel_index(np.argmax(mask, axis=None), shape=(512,64))
    return kp[1], kp[0]

def findCoordinates(mask):

    hm_sum = np.sum(mask)

    index_map = [j for i in range(512) for j in range(64)]
    index_map = np.reshape(index_map, newshape=(512,64))
    index_map2 = [i for i in range(512) for j in range(64)]
    index_map2 = np.reshape(index_map2, newshape=(512,64))
    x_score_map = mask * index_map / hm_sum
    y_score_map = mask * index_map2 / hm_sum

    px = np.sum(np.sum(x_score_map, axis=None))
    py = np.sum(np.sum(y_score_map, axis=None))

    return px, py


def calcKeypoints(model, gen):
    kps_gt = []
    kps_preds = []
    nbatches = len(gen)

    for i in range(nbatches+1):
        # print("\nBatch {}".format(i))
        imgs, batch_gt = gen[i]
        batch_preds = model.predict_on_batch(imgs)
        n_imgs = imgs.shape[0]
        # print("\t# of Images {}".format(n_imgs))
        for j in range(n_imgs):
            mask_gt = batch_gt[j]
            mask_gt = np.reshape(mask_gt, newshape=(512, 64, 1))
            mask_pred = batch_preds[j]
            mask_pred = np.reshape(mask_pred, newshape=(512, 64, 1))
            nchannels = mask_gt.shape[-1]
            # print(nchannels)
            gt_list = []
            pred_list = []

            for k in range(nchannels):
                xgt, ygt = maskToKeypoints(mask_gt[:, :, k]) # maskToKeypoints(mask_gt[:, :, k])
                xpred, ypred = maskToKeypoints(mask_pred[:, :, k])  # maskToKeypoints(mask_pred[:, :, k])

                gt_list.append(xgt)
                gt_list.append(ygt)

                pred_list.append(xpred)
                pred_list.append(ypred)
            kps_gt.append(gt_list)
            kps_preds.append(pred_list)
    

    return np.array(kps_gt, dtype=np.float32), np.array(kps_preds, dtype=np.float32)


def calcRMSError(kps_gt, kps_preds):

    N = kps_gt.shape[0] * (kps_gt.shape[-1] // 2)
    error = np.sqrt(np.sum((kps_gt-kps_preds)**2)/N)

    return error

def show_keypoints(batch_imgs, batch_labels, predictions=None):

    def draw_keypoints(img, keypoints, col):
        # print("\n{}".format(len(keypoints)))
        for i in range(0, len(keypoints)-1, 2):
            # print(i)
            kpx = int(keypoints[i])
            kpy = int(keypoints[i+1])
            img = cv.circle(img, center=(kpx,kpy), radius=2, color=col, thickness=2)

        return img

    for i in range(len(batch_imgs)):

        img = batch_imgs[i]
        img = np.reshape(img, newshape=(512, 64))
        img = np.stack([img,img,img], axis=-1)

        # draw ground-truth keypoints on image
        if batch_labels is not None:
            img = draw_keypoints(img, batch_labels[i], col=(0,0,255))

        # draw predicted keypoints on image
        if predictions is not None:
            img = draw_keypoints(img, predictions[i], col=(255,0,0))
        
        cv.imshow('result', img)
        while True:
            if cv.waitKey(1000) == ord('n'):
                break
        # axes[r, c].imshow(img)
    # plt.savefig(name)
    # plt.show()

def showAllMasks(img_mask, name, nrows=3, ncols=5):

    # fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

    # r = -1
    print(img_mask.shape)
    print(img_mask.shape[-1])
    for i in range(img_mask.shape[-1]):

        img = img_mask[:, :, i]
        # print(img.shape)
        img = np.reshape(img, newshape=(512,64))
        img = np.stack([img,img,img], axis=-1)

        # c = i % ncols

        # if i % ncols == 0:
            # r += 1
        cv.imshow(name + str(i), img)
        while True:
            if cv.waitKey(1000) == ord('n'):
                break
        # axes[r, c].imshow(img)
    # plt.figure(figsize=(12,12))
    # plt.show()

def show_masks(batch_imgs, batch_gt_masks, include_preds= False, predictions=None):

    for c in range(len(batch_imgs)):
        
        # original image
        img = batch_imgs[c]
        img = np.reshape(img, newshape=(512,64))
        img = np.stack([img,img,img], axis=-1)

        # ground-truth mask
        gt_mask = batch_gt_masks[c]
        gt_mask = np.reshape(gt_mask, newshape=(512,64,1))
        gt_mask = np.sum(gt_mask, axis=-1)
        gt_mask = np.stack([gt_mask,gt_mask,gt_mask], axis=-1)


        # prediction mask
        if include_preds: 
            pred_mask = predictions[c]
            pred_mask = np.reshape(pred_mask, newshape=(512,64,1))
            pred_mask = np.sum(pred_mask, axis=-1)
            pred_mask = np.stack([pred_mask,pred_mask,pred_mask], axis=-1)
        
        concat = cv.hconcat([img, gt_mask, pred_mask])
        cv.imshow('img, gt, pred', concat)
        
        while True:
            if cv.waitKey(1000) == ord('n'):
                break

        
Nkeypoints = 1
W = 64
H = 512

def main():
    data_dir = "./data"
    train_dir = "train3"
    train_csv = "train3.csv"
    test_dir = "test3"
    test_csv = "test3.csv"

    df_train = pd.read_csv(os.path.join(train_csv))
    df_test = pd.read_csv(os.path.join(test_csv))

    n_train = df_train['name'].size
    n_test = df_test['name'].size

    df_kp = df_train.iloc[:,1:5]

    idxs = []

    img_dict = {}
    kp_dict = {}

    for i in range(n_train):

        if True in df_train.iloc[i, 1:5].isna().values:
            continue
        else:
            idxs.append(i)

            img_dict[i] = df_train['name'][i]

            # keypoints
            kp = df_kp.iloc[i].values.tolist()
            kp_dict[i] = kp

    random.shuffle(idxs)

    # subset = int(0.1*len(idxs))

    cutoff_idx = int(0.9*len(idxs))
    train_idxs = idxs[0:cutoff_idx]
    val_idxs = idxs[cutoff_idx:len(idxs)]

    print("\n# of Training Images: {}".format(len(train_idxs)))
    print("# of Val Images: {}".format(len(val_idxs)))

    transform_dict = {"Flip": False, "Shift": False, "Scale": False, "Rotate": False}

    train_gen = MaskGenerator(os.path.join(data_dir, train_dir),
                                train_idxs,
                                img_dict,
                                kp_dict,
                                transform_dict=transform_dict,
                                augment=False, 
                                batch_size=1)

    val_gen = MaskGenerator(os.path.join(data_dir, train_dir),
                                val_idxs,
                                img_dict,
                                kp_dict,
                                augment=False,
                                batch_size=8)
    imgs, masks = val_gen[0]
    unet = UNET(input_shape=(512, 64, 1))
    unet.load_weights("./unet_lr=1e-06.h5")
    preds = unet.predict_on_batch(imgs)
    kps_gt, kps_preds = calcKeypoints(unet, val_gen)
    pred_mask = preds[1]
    pred_mask = np.reshape(pred_mask, newshape=(512,64,1))
    # showAllMasks(pred_mask, "test")
    # showAllMasks(preds[1], "name")
    # print(kps_gt.shape, kps_preds.shape)
    rms_error = calcRMSError(kps_gt, kps_preds)
    print("Validation RMS Error = {}".format(rms_error))
    show_masks(imgs, masks, include_preds=True, predictions = preds)
    show_keypoints(imgs, kps_gt, predictions=kps_preds)

if __name__ == "__main__":
    main()