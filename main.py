import streamlit as st
import numpy as np
import cv2 
import os 


import os
import pandas as pd

import os
import cv2
import pandas as pd
import torch
import torchvision
from torchvision import io
from torch.utils.data import DataLoader, Dataset, random_split
from torch import nn
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn.functional as F
#from torchsummary import summary
import torchaudio

from collections import OrderedDict
import math, random
from torchaudio import transforms
from IPython.display import Audio
from sklearn.model_selection import train_test_split
import numpy as np

import torch
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super().__init__()
        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

model =UNet().to(device)
#print(model)
model.load_state_dict(torch.load("saved_model.pth", map_location=torch.device('cpu')))

class InfeDS(Dataset):
    def __init__(self, X):
        self.X = X
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        image_file = os.path.join(self.X[idx])
        image = cv2.imread(image_file)
        image = cv2.resize(image, (224, 224))
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        return image
        
        
        
        
def Prediction(image_path, model):
    file_paths = []
    masks = []
    file_paths.append(image_path)
    X_infe = np.array(file_paths)
    infe_data = InfeDS(X_infe)
    infe_dataloader = DataLoader(infe_data, batch_size=1)

    for X in infe_dataloader:
        X = X.to(torch.float32)
        X = X.to(device)
        y_pred = model(X)
        return y_pred


uploaded_file = st.file_uploader("Upload ảnh MRI não bộ", type=["jpg", "png", "jpeg", "tif"])

if uploaded_file != None:
    byte_image = uploaded_file.read()
    temp_file = "temp_file.tif"
    with open(temp_file, 'wb') as f:
        f.write(byte_image)
    img = cv2.imread(temp_file)
    img_resize = cv2.resize(img, (224, 224)) 
    img_rgb = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
    st.image(img_rgb)
    on_click = st.button("Prediction")
    if on_click:
        with st.spinner('Wait for it...'):
            image_path = temp_file
            #mask_path = os.path.join(data_path, y_test[9])
            a = Prediction(image_path, model)
            mask = a[0][0]
            mask = mask.cpu().detach().numpy()
            seg_image = np.zeros((224, 224))
            gray_image = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
        
        #cv2.imwrite("mask.jpg", mask)
        mask1 = np.zeros((224, 224))
        for i in range(mask1.shape[0]):
            for j in range(mask1.shape[1]):
                if mask[i][j] >= 0.95:
                    mask1[i][j] = 1
                else:
                    mask1[i][j] = 0
        st.image(mask1)
            #print(mask)
            #for i in range(seg_image.shape[0]):
            #    for j in range(seg_image.shape[1]):
            #        if mask[i][j] == 1:
            #            seg_image[i][j] = 255 - gray_image[i][j]
            #            #seg_image[i][j] = 25
            #        else:
            #            seg_image[i][j] = gray_image[i][j]
        #temp_seg_file = "temp_seg_file.jpg"
        #cv2.imwrite(temp_seg_file, seg_image)
        #st.image(temp_seg_file)   
        #cv2.imwrite("mask.jpg", mask)
        #st.image("mask.jpg")
        #os.remove(temp_file)
        #os.remove(temp_seg_file)
    