import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import selfies as sf
import torch
import torch.nn as nn
import wandb

from torch.utils.data import DataLoader

from utils import SelfiesData, VAE, collate_fn, tensor2selfies, Parameters, RnnType

class Logger():
    def log(self, msg):
        print(msg)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

smiles_df = pd.read_csv("data/zinc15_250K_2D.csv")
smiles = np.random.choice(smiles_df["smiles"].values, size=1_000)
# smiles = smiles_df["smiles"].values

sfs = [sf.encoder(s) for s in smiles]

split = round(len(sfs)*0.8)
sf_train = SelfiesData(sfs[:split])
sf_valid = SelfiesData(sfs[split:])

params_dict = dict(
    batch_size = 32,
    rnn_type = RnnType.LSTM,
    vocab_size = sf_train.n_symbols,
    embed_dim = 250,
    rnn_hidden_dim = 250,
    latent_dim = 250,
    n_epochs = 50,
    learning_rate = 1e-2,
    n_layers = 2,
    bidirectional_encoder = True,
    # k = 0.00125,
    # x0 = 2500,
    # anneal_function = "logistic",
    rnn_dropout = 0.0,
    # word_keep_rate = 0.0,
    temperature = 0.9,
    temperature_min = 0.5,
    temperature_dec = 0.000002,
    grad_clip = 1.0,
)
params = Parameters(params_dict)

train_dataloader = DataLoader(sf_train, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(sf_valid, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn)

criterion = nn.NLLLoss()
vae = VAE(
    device=device,
    params=params,
    criterion=criterion,
    logger=wandb)

wandb.init(project="mol-vae", entity="rmeinl", config=params_dict)
wandb.watch([vae.encoder, vae.decoder])

total_loss = 0.0
for epoch in range(1, params.n_epochs+1):
    print(f"Epoch #{epoch}")
    
    vae.train()
    epoch_loss = vae.train_epoch(train_dataloader)

    vae.eval()
    if epoch % 5 == 0:
        target = torch.tensor(np.random.choice(train_dataloader.dataset)).to(device).unsqueeze(0)
        print(f"Target:\n {tensor2selfies(sf_train, target)}")
        generated = vae.evaluate(target, max_steps=sf_train.n_symbols)[0]
        print(f"Generated:\n {tensor2selfies(sf_train, torch.tensor(generated))}")

# it = iter(train_dataloader)
# input_tensor, input_lengths = next(it)

# encoder = EncoderRNN(input_size=sf_train.n_symbols, hidden_size=hidden_size, latent_size=latent_size)
# decoder = DecoderRNN(latent_size=latent_size, hidden_size=hidden_size, output_size=sf_train.n_symbols, word_keep_rate=0.0)

# batch_size, input_length = input_tensor.size()

# mu, logvar, z = encoder(input_tensor, input_lengths)
# outputs_tensor = decoder(z, input_tensor, input_lengths, temperature)
# outputs_tensor = outputs_tensor.data.max(1)[1]

