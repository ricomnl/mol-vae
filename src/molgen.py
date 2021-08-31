import numpy as np
import pandas as pd
import selfies as sf
import torch
import torch.nn as nn
import wandb

from torch.utils.data import DataLoader

from utils import AeType, SelfiesData, AE, collate_fn, tensor2selfies, Parameters, RnnType, selfies2image

class Logger():
    def log(self, msg):
        print(msg)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

smiles_df = pd.read_csv("data/zinc15_250K_2D.csv")
smiles = np.random.choice(smiles_df["smiles"].values, size=100)
# smiles = smiles_df["smiles"].values

sfs = [sf.encoder(s) for s in smiles]

split = round(len(sfs)*0.8)
sf_train = SelfiesData(sfs[:split])
sf_valid = SelfiesData(sfs[split:])

params_dict = dict(
    batch_size = 64,
    ae_type = AeType.VAE,
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
model = AE(
    device=device,
    params=params,
    criterion=criterion,
    logger=wandb)

wandb.init(project="mol-vae", entity="rmeinl", config=params_dict)
wandb.watch([model.encoder, model.decoder])

total_loss = 0.0
for epoch in range(1, params.n_epochs+1):
    print(f"Epoch #{epoch}")
    
    model.train()
    epoch_loss = model.train_epoch(train_dataloader)

    model.eval()
    if epoch % 5 == 0:
        target = torch.tensor(np.random.choice(train_dataloader.dataset)).unsqueeze(0)
        target_selfies = tensor2selfies(sf_train, target)
        print(f"Target:\n {target_selfies}")
        generated = model.evaluate(target, max_steps=sf_train.n_symbols)[0]
        generated_selfies = tensor2selfies(sf_train, torch.tensor(generated))
        print(f"Generated:\n {generated_selfies}")
        wandb.log({
            "ground-truth": wandb.Image(selfies2image(target_selfies), caption=target_selfies),
            "predicted": wandb.Image(selfies2image(generated_selfies), caption=generated_selfies)
        })

# it = iter(train_dataloader)
# input_tensor, input_lengths = next(it)

# encoder = EncoderRNN(input_size=sf_train.n_symbols, hidden_size=hidden_size, latent_size=latent_size)
# decoder = DecoderRNN(latent_size=latent_size, hidden_size=hidden_size, output_size=sf_train.n_symbols, word_keep_rate=0.0)

# batch_size, input_length = input_tensor.size()

# mu, logvar, z = encoder(input_tensor, input_lengths)
# outputs_tensor = decoder(z, input_tensor, input_lengths, temperature)
# outputs_tensor = outputs_tensor.data.max(1)[1]