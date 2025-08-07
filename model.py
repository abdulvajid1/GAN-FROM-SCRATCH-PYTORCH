import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Annotated


class Generator(nn.Module):
    """Generate new image from input sampled from normal distribution.
    """
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size
        
        nn.Sequential(
            
        )
    
    def forward(self, hidden_size):
        p
        
        
    
class Descriminator(nn.Module):
    """
    Classify image => real or fake
    """
    def __init__(self):
        super().__init__()
        self.descriminator = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=124, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features="don't know now", out_features=64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img_batch, labels) -> Annotated[(list, float), "()"]:
        predictions = self.descriminator(img_batch)
        return predictions
        
    
class DCGAN(nn.Module):
    """
    GAN model
    """
    def __init__(self,generator: Generator, descriminator: Descriminator, batch_size, device):
        super().__init__()
        self.generator = generator
        self.descrimator = descriminator
        self.batch_size = batch_size
        self.device = device
        pass
    
    def generate_image_batch(self, batch_size):
        pass
    
    def descriminate_batch(self, img_batch):
        pass 
    
    def forward(self, x):
        fake_img_batch = self.generate_image_batch(self.batch_size) # (B, C, W, H)
        real_img_batch = x # real image comes from the dataset
        
        real_labels = torch.ones(size=(self.batch_size, ), device=self.device)
        fake_labels = torch.zeros(size=(self.batch_size, ), device=self.device)
        
        # Descriminate
        real_desc_pred = self.descrimator(real_img_batch)
        fake_desc_pred = self.descrimator(fake_img_batch) # Descrimating Generated image
        
        # Descriminator
        real_desc_loss = F.binary_cross_entropy(real_desc_pred, real_labels)
        fake_desc_loss = F.binary_cross_entropy(fake_desc_loss, fake_labels)
        descriminator_loss = (real_desc_loss + fake_desc_loss) / 2.0 # average
        
        # Generator loss
        generator_loss = F.binary_cross_entropy(fake_desc_pred, real_labels)
        
        return generator_loss, descriminator_loss
        
        
    