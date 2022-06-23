import torch.nn as nn
import torch.nn.functional as F

class roomnet(nn.Module):
    def __init__(self):
        super(roomnet, self).__init__()
        self.conv1_1 = conv_bn_relu()
        self.conv1_2 = conv_bn_relu()
        
        self.conv2_1 = conv_bn_relu()
        self.conv2_2 = conv_bn_relu()
        
        self.conv3_1 = conv_bn_relu()
        self.conv3_2 = conv_bn_relu()
        self.conv3_3 = conv_bn_relu()
        
        self.conv4_1 = conv_bn_relu()
        self.conv4_2 = conv_bn_relu()
        self.conv4_3 = conv_bn_relu()
        
        self.conv5_1 = conv_bn_relu()
        self.conv5_2 = conv_bn_relu()
        self.conv5_3 = conv_bn_relu()
        
        self.conv6_1 = conv_bn_relu()
        self.conv6_2 = conv_bn_relu()
        self.conv6_3 = conv_bn_relu()
        
        self.conv7_1 = conv_bn_relu()
        self.conv7_2 = conv_bn_relu()
        self.conv7_3 = conv_bn_relu()
        
        self.conv8_1 = conv_bn_relu()
        self.conv8_2 = conv_bn_relu()
        
        
        self.dropout3 = nn.Dropout2d()
        self.dropout4 = nn.Dropout2d()
        self.dropout5 = nn.Dropout2d()
        self.dropout6 = nn.Dropout2d()
        
        self.fc1 = nn.Linear()
        self.fc2 = nn.Linear()
        self.fc3 = nn.Linear()
        
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten(dim=-1)
        
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.maxunpool = nn.MaxUnpool2d(2, stride=2)
        
        self.he_initialization()
    
    
    def he_initialization(self):
        # He initialization
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
        x = self.droppout3(x)        

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x, index_4 = self.maxpool(x)
        x = self.droppout4(x)  
        
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x_break, index_5 = self.maxpool(x)
        
        
        # Keypoint Decoding
        x_key = self.droppout5(x_break)
        
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