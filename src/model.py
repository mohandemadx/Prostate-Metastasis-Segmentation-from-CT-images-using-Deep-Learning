"""
File: model.py
Description: This file defines the architecture for the segmentation model. 
             It currently implements a U-Net model using PyTorch.
Author: Carole Emad
Date: October 30, 2024
Version: 1.0
Contributions:
  - 
"""

# Main Logic 
import torch.nn as nn
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric


class Unet(nn.Modulenn.Module):
    """
    input of the model: 4D tensor (nb of images, nb of channels, height, width) (D,C,H,W)
    parametrs:
    output: 4D tensor of the segmented image
    """
  
    def __init__(self, input_channels=1, output_channels =1, max_pool_ker =2, max_pool_stride=3):
        super (Unet, self).__init__()

        # Encoder part
        self.encoder1 = self.triple_conv(input_channels, 32, 0.2)
        self.pool = nn.MaxPool2d(max_pool_ker)

        self.encoder2 = self.triple_conv(32, 64, 0.2)
        
        self.encoder3 = self.triple_conv(64, 128, 0.3)
        
        self.encoder4 = self.triple_conv(128, 256, 0.3)
        
        self.encoder5 = self.triple_conv(256, 512, 0.4)

        #bottleneck
        self.bottleneck = self.double_conv(512, 1024, 0.4)

        # Decoder part
        self.upconv5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder5 = self.double_conv(1024, 512, 0.4)

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = self.double_conv(512, 256, 0.4)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = self.double_conv(256, 128, 0.4)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = self.double_conv(128, 64, 0.4)
    
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = self.double_conv(64, 32, 0.4)

        # Final 1x1 convolution for segmentation output
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.final_activation = nn.Sigmoid()

        # Initialize metrics
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction="mean")


    def triple_conv(self, in_channels, out_channels, dropout_rate):
        """Helper function to create three convolutional layers with ReLU, BatchNorm, and optional Dropout."""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
          ]
        return nn.Sequential(*layers)

    def double_conv(self, in_channels, out_channels, dropout_rate):
        """Helper function to create two convolutional layers with ReLU, BatchNorm, and optional Dropout."""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
          ]
        return nn.Sequential(*layers)
    
    def forward(self,x):
        
        #Encoder path
        enc1= self.encoder1(x)
        print(enc1.size())
        enc2= self.encoder2(self.pool(enc1))
        print(enc2.size())
        enc3= self.encoder3(self.pool(enc2))
        print(enc3.size())
        enc4= self.encoder4(self.pool(enc3))
        print(enc4.size())
        enc5= self.encoder5(self.pool(enc4))
        print(enc5.size())

        #bottleneck
        bottleneck= self.bottleneck(self.pool(enc5))
        print(bottleneck.size())

        #Decoder path
        x = self.upconv5(bottleneck)
        print(x.size())
        x= torch.cat([x, enc5], dim=1)
        print(x.size())
        dec5 = self.decoder5(x)

        dec5 = self.upconv4(dec5)
        dec5= torch.cat([dec5, enc4], dim=1)
        dec4 = self.decoder4(dec5)

        dec4 = self.upconv3(dec4)
        dec4= torch.cat([dec4, enc3], dim=1)
        dec3 = self.decoder3(dec4)

        dec3 = self.upconv2(dec3)
        dec3= torch.cat([dec3, enc2], dim=1)
        dec2 = self.decoder2(dec3)

        dec2 = self.upconv1(dec2)
        dec2= torch.cat([dec2, enc1], dim=1)
        dec1 = self.decoder1(dec2)

        # Final segmentation output
        output = self.final_activation(self.final_conv(dec1))
        return output
    
    def evaluate(self, test_loader):
        
        # Model evaluation loop over test set
        for x_batch, y_batch in test_loader:
            y_pred = model(x_batch)
            self.dice_metric(y_pred, y_batch)         
            self.hausdorff_metric(y_pred, y_batch)    

        # After the loop, get the final metric scores
        dice_score = self.dice_metric.aggregate().item()
        hausdorff_score = self.hausdorff_metric.aggregate().item()

        return {
            "Dice Coefficient": dice_score,
            "Hausdorff Distance": hausdorff_score
        }
    
#initialize the model    
model = Unet()
print(model)

random_image = torch.randn(1,1,1024,1024)
print(random_image)
out = model.forward(random_image)
print(out)

