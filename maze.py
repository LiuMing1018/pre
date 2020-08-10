#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@version: python3.6
@author: QLMX
@contact: wenruichn@gmail.com
@time: 2019-08-14 16:07
"""
import numpy as np
import pandas as pd
import streamlit as st
import cv2

import torchvision.transforms as transforms
import torch
from PIL import Image
from collections import OrderedDict
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torch import nn
import os, time
import torchvision.models as models
from resnetxt_wsl import resnext101_32x8d_wsl, resnext101_32x16d_wsl, resnext101_32x32d_wsl
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

args = {}
args['arch'] = 'resnext101_32x16d_wsl'
args['pretrained'] = False
args['num_classes'] = 10
args['image_size'] = 320


class classfication_service():
    def __init__(self, model_path):
        self.model = self.build_model(model_path)
        self.pre_img = self.preprocess_img()
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_id_name_dict = \
            {
                "0": "其他垃圾/一次性快餐盒",
                "1": "其他垃圾/污损塑料",
                "2": "其他垃圾/烟蒂",
                "3": "其他垃圾/牙签",
                "4": "其他垃圾/破碎花盆及碟碗",
                "5": "其他垃圾/竹筷",
                "6": "厨余垃圾/剩饭剩菜",
                "7": "厨余垃圾/大骨头",
                "8": "厨余垃圾/水果果皮",
                "9": "厨余垃圾/水果果肉"
            }

    def build_model(self, model_path):
        if args['arch'] == 'resnext101_32x16d_wsl':
            model = resnext101_32x16d_wsl(pretrained=False, progress=False)
        if args['arch'] == 'resnext101_32x8d':
            model = models.__dict__[args['arch']]()
        elif args['arch'] == 'efficientnet-b7':
            model = EfficientNet.from_name(args['arch'])

        layerName, layer = list(model.named_children())[-1]
        exec("model." + layerName + "=nn.Linear(layer.in_features," + str(args['num_classes']) + ")")

        if torch.cuda.is_available ():
            model = nn.DataParallel (model).cuda ()
            modelState = torch.load (model_path)
            model.load_state_dict (modelState)
            model = model.cuda ()
        else:
            # model = nn.DataParallel(model)
            modelState = torch.load (model_path, map_location='cpu')
            new_state_dict = OrderedDict ()
            for key, value in torch.load (model_path, map_location='cpu').items ():
                name = key[7:]
                if name == "fc.1.weight":
                    name = "fc.weight"
                elif name == "fc.1.bias":
                    name = "fc.bias"
                new_state_dict[name] = value
            model.load_state_dict (new_state_dict)
        return model

    def preprocess_img(self):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        infer_transformation = transforms.Compose([
            Resize((args['image_size'], args['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        return infer_transformation

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                img = Image.open(file_content)
                img = self.pre_img(img)
                preprocessed_data[k] = img
        return preprocessed_data

    def _inference(self, data):
        """
        model inference function
        Here are a inference example of resnet, if you use another model, please modify this function
        """
        img = data['input_img']
        img = img.unsqueeze(0)
        img = img.to(self.device)
        with torch.no_grad():
            pred_score = self.model(img)

        if pred_score is not None:
            _, pred_label = torch.max(pred_score.data, 1)
            result = {'result': self.label_id_name_dict[str(pred_label[0].item())]}
        else:
            result = {'result': 'predict score is None'}

        return result

    def _postprocess(self, data):
        return data


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w+w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h+h_padding))

        img = img.resize(self.size, self.interpolation)

        return img



def PRE():
    if __name__ == '__main__':
        model_path = 'E:/data0/data0/search/qlmx/clover/garbage/res_16_288_last1/' + 'model_32_9979_9361.pth'
        #    model = nn.DataParallel(model).cuda()
        infer = classfication_service (model_path)
        input_dir = 'D:/pictures/'
        files = os.listdir (input_dir)
        t1 = int (time.time () * 1000)
        index=0
        for file_name in files:
            file_path = os.path.join (input_dir, file_name)
            print (file_path)
            img = Image.open (file_path)
            img = infer.pre_img (img)
            tt1 = int (time.time () * 1000)
            result = infer._inference ({'input_img': img})
            print (result)
            image = Image.open ('D:/pictures/'+str(index)+'.jpg')
            st.image (image, caption='分割图像'+str(index), use_column_width=True)
            #   image = Image.open ('1.jpg')
            #   st.image (image, caption='分割图像2', use_column_width=True)
            st.write(result)
            index+=1


            # tt2 = int(time.time() * 1000)
            # print((tt2 - tt1) / 100)
        # t2 = int(time.time()*1000)
        # print((t2 - t1)/100)
if st.button("上传图片"):
    use_default_image = st.checkbox ("Use default image")
    uploaded_file = st.file_uploader ("Choose an image", ["jpg", "jpeg", "png"])  # image uploader，存储路径还不知道
    st.set_option('deprecation.showfileUploaderEncoding', False)
    if use_default_image:
        opencv_image = cv2.imread ("D:/picture/test2.jpg")
    elif uploaded_file:
        file_bytes = np.asarray (bytearray (uploaded_file.read ()), dtype=np.uint8)
        opencv_image = cv2.imdecode (file_bytes, 1)
        cv2.imwrite ('D:/picture/test2.jpg', opencv_image)
    img = cv2.imread ("D:/picture/test2.jpg")  # 载入图像
    img1 = Image.open ('D:/picture/test2.jpg')
    h, w = img.shape[:2]  # 获取图像的高和宽

    #cv2.namedWindow('Origin', 0)
    #cv2.resizeWindow('Origin', 682, 500)
    #cv2.imshow("Origin", img)  # 显示原始图像
    #cv2.waitKey(0)

    blured = cv2.blur (img, (5, 5))  # 进行滤波去掉噪声

    #cv2.namedWindow('Blur', 0)
    #cv2.resizeWindow('Blur', 682, 500)
    #cv2.imshow("Blur", blured)  # 显示低通滤波后的图像
    #cv2.waitKey(0)

    mask = np.zeros ((h + 2, w + 2), np.uint8)  # 掩码长和宽都比输入图像多两个像素点，满水填充不会超出掩码的非零边缘
    # 进行泛洪填充
    cv2.floodFill (blured, mask, (w - 1, h - 1), (255, 255, 255), (2, 2, 2), (3, 3, 3), 8)

    #cv2.namedWindow('floodfill', 0)
    #cv2.resizeWindow('floodfill', 682, 500)
    #cv2.imshow("floodfill", blured)
    #cv2.waitKey(0)

    # 得到灰度图
    gray = cv2.cvtColor (blured, cv2.COLOR_BGR2GRAY)

    #cv2.namedWindow('gray', 0)
    #cv2.resizeWindow('gray', 682, 500)
    #cv2.imshow("gray", gray)
    #cv2.waitKey(0)

    # 定义结构元素
    kernel = cv2.getStructuringElement (cv2.MORPH_RECT, (50, 50))
    # 开闭运算，先开运算去除背景噪声，再继续闭运算填充目标内的孔洞
    opened = cv2.morphologyEx (gray, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx (opened, cv2.MORPH_CLOSE, kernel)

    #cv2.namedWindow('closed', 0)
    #cv2.resizeWindow('closed', 682, 500)
    #cv2.imshow("closed", closed)
    #cv2.waitKey(0)

    # 求二值图
    ret, binary = cv2.threshold (closed, 250, 255, cv2.THRESH_BINARY)
    binary = 255 - binary

    contours, hierarchy = cv2.findContours (binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate (contours):
        x, y, w, h = cv2.boundingRect (contour)
        area = cv2.contourArea (contour)
        if area < 5000:
            continue
        cv2.rectangle (binary, (x - 20, y - 20), (x + w + 20, y + h + 20), (255, 255, 255), -1)

    #cv2.namedWindow('binary', 0)
    #cv2.resizeWindow('binary', 682, 500)
    #cv2.imshow("binary", binary)
    #cv2.waitKey(0)

    contours, hierarchy = cv2.findContours (binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate (contours):
        x, y, w, h = cv2.boundingRect (contour)
        area = cv2.contourArea (contour)
        if area < 3000:
            continue
        out = img1.crop ((x - 40, y - 40, x + w + 40, y + h + 40))
        root = "D:/pictures/"
        out.save (root + str (i) + '.jpg', 'PNG')
        # cv2.rectangle(binary, (x-100, y-100), (x + w +100, y + h +100), (255, 255, 255), -1)
    cv2.destroyAllWindows ()
    st.set_option ('deprecation.showfileUploaderEncoding', False)
if st.button("开始"):
    PRE()

