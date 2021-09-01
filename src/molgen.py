import numpy as np
import pandas as pd
import selfies as sf
import torch
import torch.nn as nn
import wandb

from torch.utils.data import DataLoader

from utils import AeType, SelfiesData, AE, AnnealType, collate_fn, tensor2selfies, Parameters, RnnType, selfies2image

class Logger():
    def log(self, msg):
        print(msg)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# smiles_df = pd.read_csv("data/zinc15_250K_2D.csv")
# smiles = np.random.choice(smiles_df["smiles"].values, size=10_000)
# smiles = smiles_df["smiles"].values

# sfs = [sf.encoder(s) for s in smiles]
# only keep the ones shorter than 32 tokens for now
# sfs = np.array([s for s in sfs if len(list(sf.split_selfies(s))) <= 32])
# np.save("data/lt_32_tkn_selfies_zinc15.npy", sfs)
sfs = np.load("data/lt_32_tkn_selfies_zinc15.npy")
sfs = np.random.choice(sfs, size=1_000)

split = round(len(sfs)*0.8)
sf_train = SelfiesData(sfs[:split])
sf_valid = SelfiesData(sfs[split:])

params_dict = dict(
    batch_size = 64,
    ae_type = AeType.VAE,
    rnn_type = RnnType.LSTM,
    vocab_size = sf_train.n_symbols,
    embed_dim = 1024,
    rnn_hidden_dim = 1024,
    latent_dim = 512,
    n_epochs = 50,
    learning_rate = 1e-3,
    n_layers = 2,
    bidirectional_encoder = True,
    k = 0.025,
    x0 = 250,
    anneal_function = AnnealType.LOGISTIC,
    rnn_dropout = 0.1,
    word_dropout_rate = 0.25,
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
    if epoch % 1 == 0:
        target = torch.tensor(np.random.choice(train_dataloader.dataset)).unsqueeze(0)
        target_selfies = tensor2selfies(sf_train, target)
        print(f"Target:\n {target_selfies}")
        generated = model.evaluate(target, max_steps=sf_train.n_symbols)[0]
        generated_selfies = tensor2selfies(sf_train, torch.tensor(generated))
        print(f"Generated:\n {generated_selfies}")
        wandb.log({
            "predicted": [
                wandb.Image(selfies2image(target_selfies), caption=target_selfies),
                wandb.Image(selfies2image(generated_selfies), caption=generated_selfies)
            ]
        })

# sweep_config = {
#   "name" : "sweep",
#   "method" : "bayes",
#   "parameters" : {
#     "embed_dim" : {
#       "values" : [100, 250, 500]
#     },
#     "rnn_hidden_dim" : {
#       "values" : [100, 250, 500]
#     },
#     "latent_dim" : {
#       "values" : [100, 250, 500]
#     }
#   }
# }

# sweep_id = wandb.sweep(sweep_config)

# def train():
#     with wandb.init() as run:
#         config = wandb.config
#         model = make_model(config)
#         for epoch in range(config["epochs"]):
#             loss = model.fit()  # your model training code here
#             wandb.log({"loss": loss, "epoch": epoch})

# wandb.agent(sweep_id, function=train)