#!/usr/bin/env python
# coding=UTF-8
import os
import torch
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import cv2


from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from tqdm import tqdm

# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 20, 12

# 替换成自己的配置文件
# 替换成自己的配置文件
# 替换成自己的配置文件
config_file = "/home/hs/qyl_project/maskrcnn-benchmark_kps/configs/e2e_keypoint_rcnn_R_50_FPN_1x_predict.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])


def load(img_path):
    pil_image = Image.open(img_path).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

# 根据自己的需求改
# 根据自己的需求改
# 根据自己的需求改
coco_demo = COCODemo(
    cfg,
    min_image_size=1600,
    confidence_threshold=0.7,
)

# 测试图片的路径
# 测试图片的路径
# 测试图片的路径

imgs_dir = '/home/hs/qyl_project/maskrcnn-benchmark_kps/forward/kpsForwardImgs'
img_names = os.listdir(imgs_dir)

submit_v4 = pd.DataFrame()
empty_v4 = pd.DataFrame()

filenameList = []

X1List = []
X2List = []
X3List = []
X4List = []

Y1List = []
Y2List = []
Y3List = []
Y4List = []

TypeList = []

empty_img_name = []

# for img_name in img_names:
df= pd.DataFrame(columns=['x0', 'x1', 'x2', 'x3','y0', 'y1', 'y2', 'y3'])
for i, img_name in enumerate(tqdm(img_names)):
    #print(img_name)
    path = os.path.join(imgs_dir, img_name)
    image = load(path)
    # compute predictions
    predictions = coco_demo.compute_prediction(image)
    keypoints = predictions.get_field("keypoints")
    scores = keypoints.get_field("logits")
    kps = keypoints.keypoints
    kps = torch.cat((kps[:, :, 0:2], scores[:, :, None]), dim=2).numpy()

    df.loc[img_name] = list(np.reshape(kps[0].transpose((1, 0))[:2, ],(8,)))

df['filename']=img_names
df.to_csv('/home/hs/qyl_project/maskrcnn-benchmark_kps/forward/keypoints10k.csv',index=None)

###

#xy=pd.read_csv('/home/hs/qyl_project/maskrcnn-benchmark_kps/forward/keypoints.csv')
xy=df
img_names=xy['filename']
#img_path='/home/hs/qyl_project/maskrcnn-benchmark_kps/forward/kpsForwardImgs'
for i in range(len(img_names)):
    fname=os.path.join(imgs_dir,img_names[i])
    img = cv2.imread(fname)
    
    point_size = 3
    point_color = (0, 255, 0) # BGR
    thickness = 4 # 可以为 0 、4、8
    points_list = [(int(xy['x0'][i]),int(xy['y0'][i])), (int(xy['x1'][i]),int(xy['y1'][i])), (int(xy['x2'][i]),int(xy['y2'][i])),(int(xy['x3'][i]),int(xy['y3'][i]))]
    
    for point in points_list:
	    cv2.circle(img, point, point_size, point_color, thickness)
    

    cv2.imwrite(os.path.join('/home/hs/qyl_project/maskrcnn-benchmark_kps/forward/kpsForwardResults10k',img_names[i]), img)
    


'''
    try:
        scores = predictions.get_field("scores").numpy()
        bbox = predictions.bbox[np.argmax(scores)].numpy()
        labelList = predictions.get_field("labels").numpy()
        label = labelList[np.argmax(scores)]
        print(predictions)
        print(bbox)
        filenameList.append(img_name)
        X1List.append(round(bbox[0]))
        Y1List.append(round(bbox[1]))
        X2List.append(round(bbox[2]))
        Y2List.append(round(bbox[1]))
        X3List.append(round(bbox[2]))
        Y3List.append(round(bbox[3]))
        X4List.append(round(bbox[0]))
        Y4List.append(round(bbox[3]))
        TypeList.append(label)
        # print(filenameList, X1List, X2List, X3List, X4List, Y1List,
        #       Y2List, Y3List, Y4List, TypeList)
        print(label)
    except:
        empty_img_name.append(img_name)
        print(empty_img_name)

submit_v4['filename'] = filenameList
submit_v4['X1'] = X1List
submit_v4['Y1'] = Y1List
submit_v4['X2'] = X2List
submit_v4['Y2'] = Y2List
submit_v4['X3'] = X3List
submit_v4['Y3'] = Y3List
submit_v4['X4'] = X4List
submit_v4['Y4'] = Y4List
submit_v4['type'] = TypeList

empty_v4['filename'] = empty_img_name

submit_v4.to_csv('/home/hs/qyl_project/maskrcnn-benchmark_kps/forward/submit_v4.csv', index=None)
empty_v4.to_csv('/home/hs/qyl_project/maskrcnn-benchmark_kps/forward/empty_v4.csv', index=None)
'''
    
