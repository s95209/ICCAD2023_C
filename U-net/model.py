import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, load_weights_path=None):
        super(Encoder, self).__init__()
        self.max1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, 3, padding=1)

        # Load weights if specified
        if load_weights_path:   
            self.load_state_dict(torch.load(load_weights_path))
        
    def forward(self, x):
        x0 = self.conv1(x)
        x0 = F.relu(x0)
        x1 = self.max1(x0)
        x1 = F.relu(self.conv2(x1))
        x2 = self.max1(x1)
        x2 = F.relu(self.conv3(x2))
        x3 = self.max1(x2)
        x3 = F.relu(self.conv4(x3))
        x4 = self.max1(x3)
        x4 = F.relu(self.conv5(x4))
        return x0, x1, x2, x3, x4

class Decoder(nn.Module):
    def __init__(self, load_weights_path=None):
        super(Decoder, self).__init__()
        self.conv0 = nn.ConvTranspose2d(1024, 512, 3, padding=1)
        self.conv1 = nn.ConvTranspose2d(1024, 256, 3, padding=1)
        self.conv2 = nn.ConvTranspose2d(512, 128, 3, padding=1)
        self.conv3 = nn.ConvTranspose2d(256, 64, 3, padding=1)
        self.conv4 = nn.ConvTranspose2d(128, 64, 3, padding=1)
        self.conv5 = nn.ConvTranspose2d(64, 1, 3, padding=1)
        # Load weights if specified
        if load_weights_path:
            self.load_state_dict(torch.load(load_weights_path))
            
    def forward(self, vals):
        x0 = F.relu(self.conv0(vals[4]))
        
        x1 = F.interpolate(x0, size=vals[3].size()[2:], mode='bilinear', align_corners=True)
        x1 = torch.cat((x1, vals[3]), dim=1)
        x1 = F.relu(self.conv1(x1))
        
        x2 = F.interpolate(x1, size=vals[2].size()[2:], mode='bilinear', align_corners=True)
        x2 = torch.cat((x2, vals[2]), dim=1)
        x2 = F.relu(self.conv2(x2))
        
        x3 = F.interpolate(x2, size=vals[1].size()[2:], mode='bilinear', align_corners=True)
        x3 = torch.cat((x3, vals[1]), dim=1)
        x3 = F.relu(self.conv3(x3))
        
        x4 = F.interpolate(x3, size=vals[0].size()[2:], mode='bilinear', align_corners=True)
        x4 = torch.cat((x4, vals[0]), dim=1)
        x4 = F.relu(self.conv4(x4))
        
        x5 = F.relu(self.conv5(x4))
        return x5

class Autoencoder(nn.Module):
    def __init__(self, encoder_weights=None, decoder_weights=None):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(load_weights_path=encoder_weights)
        self.decoder = Decoder(load_weights_path=decoder_weights)

    def forward(self, x):
        vals = self.encoder(x)
        x = self.decoder(vals)
        return x
