import sys
sys.path.append('PyTorch-VAE')

from experiment import VAEXperiment

class VAEExperiment(VAEXperiment):
    def __init__(self, model, exp_params):
        super().__init__(model, exp_params)
    
    def training_step(self, batch, batch_idx):
        super().training_step(batch, batch_idx, 0)

    def validation_step(self, batch, batch_idx):
        super().validation_step(batch, batch_idx, 0)