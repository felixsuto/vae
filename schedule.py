import sys 
sys.path.append('PyTorch-VAE/')

from models import BaseVAE
from experiment import VAEXperiment

class VAExperiment(VAEXperiment):
    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super().__init__(vae_model, params)
    
    def training_step(self, batch, batch_idx):
        return super().training_step(batch, batch_idx, 0)
    
    def validation_step(self, batch, batch_idx):
        return super().validation_step(batch, batch_idx, 0)