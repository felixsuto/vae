import os
import gdown
import sys
import torch
import yaml
import argparse
from data import prepare_dataloaders
import numpy as np

import torchvision.utils as vutils
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append('PyTorch-VAE')
from models import HVAE, LVAE, SWAE

vae_models = {'HVAE': HVAE, 'LVAE': LVAE, 'SWAE': SWAE}

def train(model, train_loader, val_loader, test_loader, optimizer, config, device='cuda:0'):
    model.train()
    best_epoch = 0
    best_loss = np.inf

    patience = config['patience']
    max_epochs = config['max_epochs']
    kld = config['kld_weight']
          
    for epoch in range(max_epochs):
        total_loss = 0.0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            results = model(data, labels = labels)
            loss_dict = model.loss_function(*results, M_N = kld)
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader.dataset)
        print(f'====> Epoch: {epoch + 1} Average train loss: {avg_loss:.6f}')

        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                results = model(data, labels = labels)
                loss_dict = model.loss_function(*results, M_N = kld)
                val_loss += loss_dict['loss'].item()
                
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        print(f'====> Epoch: {epoch + 1} Average validation loss: {avg_val_loss:.6f}')

        visualize(model, test_loader, epoch, config['save_dir'], device=device)

        if avg_val_loss < best_loss:
            best_epoch = epoch
            best_loss = avg_val_loss
            torch.save(model.state_dict(), config['best_path'])
            print(f'Saved model at epoch {epoch + 1}')

        if epoch - best_epoch > patience:
            print(f'Validation loss has not improved for {patience} epochs. Early stopping...')
            break
            
        model.train()

    print("Training finished.")

def visualize(model, test_loader, epoch, save_dir, device='cuda:0'):
    model.eval()
    with torch.no_grad():
        data, labels = next(iter(test_loader))
        data, labels = data.to(device), labels.to(device)
        results = model(data, labels = labels)
        recons = results[0]
    recons = recons.cpu()
    recons = (recons + 1) / 2
    vutils.save_image(recons, f"{save_dir}/reconstruction_e{epoch+1}.png", normalize=True)

def main(config):
    Path(config['trainer_params']['save_dir']).mkdir(exist_ok=True, parents=True)
    device = 'cuda:0'
    model = vae_models[config['model_params']['name']](**config['model_params']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = config['exp_params']['LR'], weight_decay=config['exp_params']['weight_decay'])
    train_loader, val_loader, test_loader = prepare_dataloaders(**config['data_params'])
    train(model, train_loader, val_loader, test_loader, optimizer, config['trainer_params'])

def parse_args():
    parser = argparse.ArgumentParser(description='VAE projectwork')
    parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='default.yaml')
    args = parser.parse_args()

    with open(args.filename, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    main(parse_args())