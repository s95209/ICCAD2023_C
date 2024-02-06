import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, load_weights_path=None):
        super(Encoder, self).__init__()
        self.max1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, 5, padding=2)
        # self.max1 = nn.MaxPool2d(2, stride=2, padding=0)
        # self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        # self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        # self.conv3 = nn.Conv2d(32, 16, 5, padding=2)

        # Load weights if specified
        if load_weights_path:   
            self.load_state_dict(torch.load(load_weights_path))
        
    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.max1(x0)
        x1 = F.relu(self.conv2(x1))
        x2 = self.max1(x1)
        x2 = F.relu(self.conv3(x2))
        x3 = self.max1(x2)
        return x0, x1, x2, x3

class Decoder(nn.Module):
    def __init__(self, load_weights_path=None):
        super(Decoder, self).__init__()
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv0 = nn.ConvTranspose2d(16, 16, 7, padding=3)
        self.conv1 = nn.ConvTranspose2d(32, 32, 7, padding=3)
        self.conv2 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.conv3 = nn.ConvTranspose2d(128, 1, 3, padding=1)
        # self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        # self.conv0 = nn.ConvTranspose2d(16, 16, 7, padding=3)
        # self.conv1 = nn.ConvTranspose2d(32, 32, 7, padding=3)
        # self.conv2 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        # self.conv3 = nn.ConvTranspose2d(128, 1, 3, padding=1)
        # Load weights if specified
        if load_weights_path:
            self.load_state_dict(torch.load(load_weights_path))
            
    def forward(self, vals):
        x1 = F.relu(self.conv0(vals[3]))
        x1 = self.up1(x1)
        x1 = F.interpolate(x1, size=(vals[2].shape[2], vals[2].shape[3]), mode='nearest')
        x1 = torch.cat((x1, vals[2]), dim=1)
        
        x2 = F.relu(self.conv1(x1))
        x2 = self.up1(x2)
        x2 = F.interpolate(x2, size=(vals[1].shape[2], vals[1].shape[3]), mode='nearest')
        x2 = torch.cat((x2, vals[1]), dim=1)

        x3 = F.relu(self.conv2(x2))
        x3 = self.up1(x3)
        x3 = F.interpolate(x3, size=(vals[0].shape[2], vals[0].shape[3]), mode='nearest')
        x3 = torch.cat((x3, vals[0]), dim=1)
        
        x4 = F.relu(self.conv3(x3))
        return x4

class Autoencoder(nn.Module):
    def __init__(self, encoder_weights=None, decoder_weights=None):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(load_weights_path=encoder_weights)
        self.decoder = Decoder(load_weights_path=decoder_weights)

    def forward(self, x):
        vals = self.encoder(x)
        x = self.decoder(vals)
        return x
