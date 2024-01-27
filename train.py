import sys
sys.path.append('PyTorch-VAE')
import os
import yaml
import argparse
import numpy as np
import pytorch_lightning as pl

from data import DataModule
from vae_experiment import VAEExperiment

from pathlib import Path
from dataset import VAEDataset

from models.vanilla_vae import VanillaVAE
from models.hvae import HVAE
from models.lvae import LVAE

vae_models = {'VanillaVAE': VanillaVAE,
              'HVAE': HVAE,
              'LVAE': LVAE }

def main(config):
    # seed for reproducibility
    #pl.utilities.seed.seed_everything(seed=config['seed'])

    # logging
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                             name=config['model_params']['name'])
    
    # data
    data = VAEDataset(**config["data_params"], pin_memory=False)
    data.setup()

    # model
    model = vae_models[config['model_params']['name']](**config['model_params'])

    # training
    trainer = pl.Trainer(logger=tb_logger,
                         callbacks=[pl.callbacks.LearningRateMonitor(),
                                    pl.callbacks.ModelCheckpoint(save_top_k=1, 
                                                                 dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                                                 monitor= "val_loss",
                                                                 save_last= True)],
                         #strategy=pl.plugins.DDPPlugin(find_unused_parameters=False),
                         **config['trainer_params'])
    
    experiment = VAEExperiment(model,
                          config['exp_params'])
    
    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    # training
    print(f"======= Training {config['model_params']['name']} =======")
    trainer.fit(experiment, datamodule=data)

def parse_args():
    parser = argparse.ArgumentParser(description='VAE projectwork')
    parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/default.yaml')
    args = parser.parse_args()

    with open(args.filename, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    main(parse_args())
