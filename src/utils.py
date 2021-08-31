import random

import numpy as np
import matplotlib.pyplot as plt
import selfies as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit.Chem import MolFromSmiles
from rdkit.Chem import Draw
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TOK_XX:
    """Class to keep order, values and ds of special tokens. If you want to change values of special tokens id,
    change it order in TOK_XX. Order is same because vocabulary adds these tokens to the beginning.
    """
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"
    PAD = "<pad>"
    TOK_XX = [UNK, PAD, BOS, EOS]
    TOK_XX_ids = {k: v for v, k in enumerate(TOK_XX)}
    UNK_id = TOK_XX_ids[UNK]
    PAD_id = TOK_XX_ids[PAD]
    BOS_id = TOK_XX_ids[BOS]
    EOS_id = TOK_XX_ids[EOS]

class SelfiesData(Dataset):
    def __init__(self, symbols):
        self.symbols = symbols
        self.TOK_XX = TOK_XX
        self.symbol2index = self.TOK_XX.TOK_XX_ids.copy()
        self.n_symbols = len(self.TOK_XX.TOK_XX)
        vocab = sf.get_alphabet_from_selfies(symbols)
        for v in vocab:
            self.symbol2index[v] = self.n_symbols
            self.n_symbols += 1
        self.index2symbol = {i:s for s,i in self.symbol2index.items()}
        self.max_length = max([len(s) for s in symbols])

    def __len__(self):
        return len(self.symbols)

    def __getitem__(self, idx):
        return self.indexes_from_selfie(self.symbols[idx])

    def indexes_from_selfie(self, selfie):
        """Numericalizes and adds BOS and EOS tokens."""
        return [self.symbol2index[s] for s in sf.split_selfies(selfie)]

    def tensor_from_selfie(self, selfie):
        indexes = self.indexes_from_selfie(selfie)
        return torch.tensor(indexes, dtype=torch.long, device=device)


def tensor2selfies(lang, tensor):
    """Convert a tensor into a selfies string."""
    return "".join([lang.index2symbol[t.item()] for t in tensor.squeeze(0) if t.item() not in (TOK_XX.TOK_XX_ids.values())])


def selfies2image(s):
    """Convert a selfies string into a PIL image."""
    mol = MolFromSmiles(sf.decoder(s), sanitize=True)
    return Draw.MolToImage(mol)


def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == "logistic":
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == "linear":
        return min(1, step/x0)


def plot_kl_loss_weight(losses, weights):
    fig, ax1 = plt.subplots()

    x = np.arange(len(losses))
    ax2 = ax1.twinx()
    ax1.plot(x, weights, "g-")
    ax2.plot(x, losses, "b-")

    ax1.set_xlabel("Step")
    ax1.set_ylabel("KL Term Weight")
    ax2.set_ylabel("KL Term Value")

    fig.show()


class Parameters:
    def __init__(self, data_dict):
        for k, v in data_dict.items():
            exec("self.%s=%s" % (k, v))


class RnnType:
    GRU = 1
    LSTM = 2


class EncoderRNN(nn.Module):
    def __init__(self, device, params, embedding):
        super(EncoderRNN, self).__init__()
        self.device = device
        self.params = params
        # Embedding layer
        self.embedding = embedding
        # RNN layer
        self.num_directions = 2 if self.params.bidirectional_encoder == True else 1
        if self.params.rnn_type == RnnType.GRU:
            self.num_hidden_states = 1
            rnn = nn.GRU
        elif self.params.rnn_type == RnnType.LSTM:
            self.num_hidden_states = 2
            rnn = nn.LSTM
        self.rnn = rnn(
            self.params.embed_dim,
            self.params.rnn_hidden_dim,
            self.params.n_layers,
            bidirectional=self.params.bidirectional_encoder,
            dropout=self.params.rnn_dropout,
            batch_first=True)
        # Initialize hidden state
        self.hidden = None
        self.hidden_factor = self.num_directions * self.params.n_layers
        # TODO: switch back to mean and logv
        # self.hidden_to_mean = nn.Linear(self.params.rnn_hidden_dim * self.hidden_factor, self.params.latent_dim)
        # self.hidden_to_logv = nn.Linear(self.params.rnn_hidden_dim * self.hidden_factor, self.params.latent_dim)

        self._init_weights()

    def _sample(self, mean, logv):
        std = torch.exp(0.5 * logv)
        # torch.randn_like() creates a tensor with values samples from N(0,1) and std.shape
        eps = torch.randn_like(std)
        # Sampling from Z~N(μ, σ^2) = Sampling from μ + σX, X~N(0,1)
        z = mean + std * eps
        return z

    def forward(self, inputs):
        # inputs: bs x sl
        batch_size, _ = inputs.size()
        # 1 x bs x n_h
        embedded = self.embedding(inputs)
        # embedded: bs x sl x n_h
        # packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        # packed[0]: bs x sl - padded x n_h
        _, self.hidden = self.rnn(embedded, self.hidden)
        # hidden: 1 x bs x n_h
        embedded = self._flatten_hidden(self.hidden, batch_size)
        # embedded: bs x n_h

        # mean = self.hidden_to_mean(embedded)
        # mu: 1 x bs x n_l
        # logv = self.hidden_to_logv(embedded)
        # logv: 1 x bs x n_l
        # Reparameterize
        # z = self._sample(mean, logv)
        # z: bs x sl x n_l
        # return mean, logv, z
        return None, None, embedded

    def init_hidden(self, batch_size=1):
        if isinstance(self.rnn, nn.modules.rnn.GRU):
            return torch.zeros(self.params.n_layers * self.num_directions, batch_size, self.params.rnn_hidden_dim).to(self.device)
        elif isinstance(self.rnn, nn.modules.rnn.LSTM):
            return (torch.zeros(self.params.n_layers * self.num_directions, batch_size, self.params.rnn_hidden_dim).to(self.device),
                    torch.zeros(self.params.n_layers * self.num_directions, batch_size, self.params.rnn_hidden_dim).to(self.device))

    def _flatten_hidden(self, h, batch_size):
        if h is None:
            return None
        elif isinstance(h, tuple): # LSTM
            x = torch.cat([self._flatten(h[0], batch_size), self._flatten(h[1], batch_size)], 1)
        else: # GRU
            x = self._flatten(h, batch_size)
        return x

    def _flatten(self, h, batch_size):
        # (num_layers*num_directions, batch_size, hidden_dim)  ==>
        # (batch_size, num_directions*num_layers, hidden_dim)  ==>
        # (batch_size, num_directions*num_layers*hidden_dim)
        return h.transpose(0, 1).contiguous().view(batch_size, -1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.001, 0.001)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

class DecoderRNN(nn.Module):
    def __init__(self, device, params, embedding, criterion):
        super(DecoderRNN, self).__init__()
        self.device = device
        self.params = params
        self.criterion = criterion
        # Embedding layer
        self.embedding = embedding
        # RNN layer
        self.num_directions = 2 if params.bidirectional_encoder == True  else 1
        if self.params.rnn_type == RnnType.GRU:
            self.num_hidden_states = 1
            rnn = nn.GRU
        elif self.params.rnn_type == RnnType.LSTM:
            self.num_hidden_states = 2
            rnn = nn.LSTM
        self.rnn = rnn(
            self.params.embed_dim,
            self.params.rnn_hidden_dim*self.num_directions,
            self.params.n_layers,
            dropout=self.params.rnn_dropout,
            batch_first=True)

        # self.hidden_factor = self.num_directions * params.n_layers
        # self.latent_to_hidden = nn.Linear(params.latent_dim, self.params.rnn_hidden_dim * self.hidden_factor)

        self.out = nn.Linear(self.params.rnn_hidden_dim * self.num_directions, self.params.vocab_size)
        self._init_weights()

    def forward(self, inputs, z, temperature, return_outputs=False):
        batch_size, sequence_length = inputs.size()

        # TODO: switch back
        # hidden = self.latent_to_hidden(z)
        x = z
        # hidden: bs x n_h
        hidden = self._unflatten_hidden(x, batch_size)
        # Restructure shape of hidden state to accommodate bidirectional encoder (decoder is unidirectional)
        hidden = self._init_hidden_state(hidden)
        # hidden: 1 x bs x n_h
        input = torch.LongTensor([[TOK_XX.BOS_id]] * batch_size).to(self.device)
        # decoder_input: bs x 1
        use_teacher_forcing = random.random() < temperature

        loss = 0
        outputs = torch.zeros((batch_size, sequence_length), dtype=torch.long).to(self.device)
        # outputs: sl x bs x n_o
        for i in range(sequence_length):
            # TODO: implement random word drop
            # drop words randomly
            # p_drop_word = torch.rand(1).item()
            # if p_drop_word > self.word_keep_rate:
            #     decoder_input = to_var(torch.LongTensor([TOK_XX.UNK_id]))
            output, hidden = self._step(input, hidden)

            _, topi = output.topk(1)
            outputs[:, i] = topi.detach().squeeze()
            loss += self.criterion(output, inputs[:, i])

            if use_teacher_forcing:
                input = inputs[:, i].unsqueeze(dim=1)
            else:
                input = topi.detach()
                if input[0].item() == TOK_XX.EOS_id:
                    break

        # print(f"In: {inputs}\nOut: {outputs}")

        # Return loss
        if return_outputs:
            return loss, outputs
        else:
            return loss

    def generate(self, z, max_steps):
        decoded_sequence = []
        # TODO: switch batch
        # hidden = self.latent_to_hidden(z)
        x = z
        # Unflatten hidden state for GRU or LSTM        
        hidden = self._unflatten_hidden(x, 1)
        # Restructure shape of hidden state to accommodate bidirectional encoder (decoder is unidirectional)
        hidden = self._init_hidden_state(hidden)
        # Create BOS token tensor as first input for decoder
        input = torch.LongTensor([[TOK_XX.BOS_id]]).to(self.device)

        for i in range(max_steps):
            output, hidden = self._step(input, hidden)
            _, topi = output.data.topk(1)

            if (topi.item() == TOK_XX.EOS_id):
                break
            else:
                decoded_sequence.append(topi.item())
                input = topi.detach()

        return decoded_sequence

    def _step(self, input, hidden):
        # output, hidden = self.gru(output, hidden)
        embedded = self.embedding(input)
        # output: bs x 1 x n_h
        output, hidden = self.rnn(embedded, hidden)
        # output: bs x 1 x n_h
        # hidden: 1 x bs x n_h
        # output = self.out(output.squeeze(dim=1))
        output = F.log_softmax(self.out(output.squeeze(dim=1)), dim=1)
        # output: bs x n_o
        return output, hidden

    def _init_hidden_state(self, encoder_hidden):
        if encoder_hidden is None:
            return None
        elif isinstance(encoder_hidden, tuple): # LSTM
            return tuple([self._concat_directions(h) for h in encoder_hidden])
        else: # GRU
            return self._concat_directions(encoder_hidden)

    def _concat_directions(self, hidden):
        # hidden.shape = (num_layers * num_directions, batch_size, hidden_dim)
        if self.num_directions > 1:
            hidden = torch.cat([hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]], 2)
            # Alternative approach (same output but easier to understand)
            #h = hidden.view(self.n_layers, self.num_directions, hidden.size(1), self.rnn_hidden_dim)
            #h_fwd = h[:, 0, :, :]
            #h_bwd = h[:, 1, :, :]
            #hidden = torch.cat([h_fwd, h_bwd], 2)
        return hidden

    def _unflatten_hidden(self, x, batch_size):
        if x is None:
            return None
        elif isinstance(self.rnn, nn.modules.LSTM):  # LSTM
            x_split = torch.split(x, int(x.shape[1]/2), dim=1)
            h = (self._unflatten(x_split[0], batch_size), self._unflatten(x_split[1], batch_size))
        else:  # GRU
            h = self._unflatten(x, batch_size)
        return h

    def _unflatten(self, x, batch_size):
        # (batch_size, num_directions*num_layers*hidden_dim)    ==>
        # (batch_size, num_directions * num_layers, hidden_dim) ==>
        # (num_layers * num_directions, batch_size, hidden_dim) ==>
        return x.view(batch_size, self.params.n_layers * self.num_directions, self.params.rnn_hidden_dim).transpose(0, 1).contiguous()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.001, 0.001)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

class VAE():
    def __init__(self, device, params, criterion, logger=None):
        self.device = device
        self.params = params
        self.criterion = criterion

        self.embedding = nn.Embedding(self.params.vocab_size, self.params.embed_dim, padding_idx=TOK_XX.PAD_id)
        self.encoder = EncoderRNN(
            device=device, 
            params=params,
            embedding=self.embedding)
        self.decoder = DecoderRNN(
            device=device,
            params=params,
            embedding=self.embedding, 
            criterion=self.criterion)
        self.encoder.to(device)
        self.decoder.to(device)

        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=params.learning_rate)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=params.learning_rate)

        self.temperature = params.temperature
        self.print_every = 1

        self.logger = logger

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def train_epoch(self, dataloader, verbose=True):
        dataset_length = len(dataloader.dataset)
        epoch_loss = 0.0
    
        for i, (inputs, _) in enumerate(dataloader):
            # Get size of batch (can differ between batches due to bucketing)
            batch_size = inputs.shape[0]

            # Add EOS token to all sequences in that batch
            eos = np.array([TOK_XX.EOS_id]*batch_size)
            inputs = np.concatenate((inputs, eos.reshape(-1, 1)), axis=1)

            # Convert to tensors and move to device
            inputs = torch.tensor(inputs).to(self.device)

            # Train batch and get batch loss
            batch_loss = self.train_batch(inputs)

            epoch_loss += batch_loss

            if verbose and i % self.print_every == 0:
                print(f"loss: {batch_loss:>5f}  [{i*len(inputs):>5d}/{dataset_length:>5d}]")
                if self.logger:
                    self.logger.log({"loss": batch_loss})

        return epoch_loss

    def train_batch(self, inputs):
        batch_size, num_steps = inputs.size()

        self.encoder.hidden = self.encoder.init_hidden(batch_size)

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        mean, logv, z = self.encoder(inputs)
        ce_loss = self.decoder(inputs, z, self.temperature)

        if self.temperature > self.params.temperature_min: 
            self.temperature -= self.params.temperature_dec

        # kld_loss = (-0.5 * torch.sum(1. + logv - mean.pow(2) - logv.exp(), 1)).mean()
        # kld_weight = kl_anneal_function(anneal_function, step, k, x0)
        # kld_weight = 0.1
        # loss = ce_loss + (kld_weight * kld_loss)
        loss = ce_loss

        # if self.logger:
        #     self.logger.log({"kld_loss": kld_loss, "kld_weight": kld_weight, "ce_loss": ce_loss, "temperature": self.temperature})

        loss.backward()
        _ = nn.utils.clip_grad_norm_(self.encoder.parameters(), self.params.grad_clip)
        _ = nn.utils.clip_grad_norm_(self.decoder.parameters(), self.params.grad_clip)
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item() / num_steps

    def evaluate(self, input, max_steps=100, use_mean=False):
        batch_size, _ = input.shape
        # Initialize hidden state
        self.encoder.hidden = self.encoder.init_hidden(batch_size)

        mean, logv, z = self.encoder(input)
        if use_mean:
            decoded_sequence = self.decoder.generate(mean, max_steps=max_steps)
            return decoded_sequence, mean
        else:
            decoded_sequence = self.decoder.generate(z, max_steps=max_steps)
            return decoded_sequence, z

def to_padded_tensor(sequences, pad_end=True, pad_idx=0, transpose=False):
    """Turns sequences of token ids into tensor with padding and optionally transpose"""
    lengths = torch.tensor([len(seq) for seq in sequences])
    max_len = max(lengths)
    tensor = torch.zeros(len(sequences), max_len).long() + pad_idx
    for i, toks in enumerate(sequences):
        if pad_end:
            tensor[i, 0:len(toks)] = torch.tensor(toks)
        else:
            tensor[i, -len(toks):] = torch.tensor(toks)
    if transpose:
        tensor = tensor.transpose(0, 1)
    return tensor, lengths

def collate_fn(data):
    data.sort(key=lambda x: len(x), reverse=True)
    input_tensor, input_lengths = to_padded_tensor(data)
    return input_tensor, input_lengths