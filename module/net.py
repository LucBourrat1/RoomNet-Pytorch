from natsort import as_ascii
import torch.nn as nn
import torch.nn.functional as F

def conv_bn_relu(in_channel, out_channel, kernel_size, stride, padding):
    conv_layer = nn.Sequential(*[nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                                 nn.BatchNorm2d(out_channel),
                                 nn.ReLU()])
    return conv_layer

class roomnet(nn.Module):
    def __init__(self):
        super(roomnet, self).__init__()
        self.conv1_1 = conv_bn_relu(3, 64, 3, 1, 1)
        self.conv1_2 = conv_bn_relu(64, 64, 3, 1, 1)
        
        self.conv2_1 = conv_bn_relu(64, 128, 3, 1, 1)
        self.conv2_2 = conv_bn_relu(128, 128, 3, 1, 1)
        
        self.conv3_1 = conv_bn_relu(128, 256, 3, 1, 1)
        self.conv3_2 = conv_bn_relu(256, 256, 3, 1, 1)
        self.conv3_3 = conv_bn_relu(256, 256, 3, 1, 1)
        
        self.conv4_1 = conv_bn_relu(256, 512, 3, 1, 1)
        self.conv4_2 = conv_bn_relu(512, 512, 3, 1, 1)
        self.conv4_3 = conv_bn_relu(512, 512, 3, 1, 1)
        
        self.conv5_1 = conv_bn_relu(512, 512, 3, 1, 1)
        self.conv5_2 = conv_bn_relu(512, 512, 3, 1, 1)
        self.conv5_3 = conv_bn_relu(512, 512, 3, 1, 1)
        
        self.conv6_1 = conv_bn_relu(512, 512, 3, 1, 1)
        self.conv6_2 = conv_bn_relu(512, 512, 3, 1, 1)
        self.conv6_3 = conv_bn_relu(512, 512, 3, 1, 1)
        
        self.conv7_1 = conv_bn_relu(512, 512, 3, 1, 1)
        self.conv7_2 = conv_bn_relu(512, 512, 3, 1, 1)
        self.conv7_3 = conv_bn_relu(512, 256, 3, 1, 1)
        
        self.conv8_1 = conv_bn_relu(256, 64, 1, 1, 0)
        self.conv8_2 = conv_bn_relu(64, 48, 1, 1, 0)
        
        
        self.dropout3 = nn.Dropout2d(p=0.5)
        self.dropout4 = nn.Dropout2d(p=0.5)
        self.dropout5 = nn.Dropout2d(p=0.5)
        self.dropout6 = nn.Dropout2d(p=0.5)
        
        self.fc1 = nn.Linear(512 * 10 * 10, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 11)
        
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()
        
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.maxunpool = nn.MaxUnpool2d(2, stride=2)
        
        self.he_initialization()
    
    
    def he_initialization(self):
        # TODO: He initialization
        pass
        
        
    def forward(self, input):
        # Encoding
        x = self.conv1_1(input)
        x = self.conv1_2(x)
        x, _ = self.maxpool(x)
        
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x, _ = self.maxpool(x)
        
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x, _ = self.maxpool(x)
        x = self.dropout3(x)        

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x, index_4 = self.maxpool(x)
        x = self.dropout4(x)  
        
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x_break, index_5 = self.maxpool(x)
        
        
        # Keypoint Decoding
        x_key = self.dropout5(x_break)
        
        x_key = self.maxunpool(x_key, index_5)
        x_key = self.conv6_1(x_key)
        x_key = self.conv6_2(x_key)
        x_key = self.conv6_3(x_key)
        x_key = self.dropout6(x_key)
        
        x_key = self.maxunpool(x_key, index_4)
        x_key = self.conv7_1(x_key)
        x_key = self.conv7_2(x_key)
        x_key = self.conv7_3(x_key)
        
        x_key = self.conv8_1(x_key)
        x_key = self.conv8_2(x_key)
        x_key = self.sigmoid(x_key)

        
        # Classification Decoding
        x_cls = self.flatten(x_break)
        x_cls = self.fc1(x_cls)
        x_cls = self.fc2(x_cls)
        x_cls = self.fc3(x_cls)
        return x_cls, x_key