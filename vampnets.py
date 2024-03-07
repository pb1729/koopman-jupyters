import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vamp_score import Affine, vamp_score


# hyperparameters:
width = 512


def res_init(module):
    """ initialization for ResLayer """
    with torch.no_grad():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None: module.bias.data.fill_(0.0)

class ResLayer(nn.Module):
    def __init__(self, dim):
        super(ResLayer, self).__init__()
        self.dim = dim
        self.layers = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.2),
            nn.Linear(dim, dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(dim, dim, bias=False),
        )
        self.apply(res_init)
    def forward(self, x):
        return x + self.layers(x)

def res_layer_weight_decay(model, coeff=0.1):
    """ weight decay only for weights in ResLayer's """
    params = []
    for m in model.modules():
        if isinstance(m, ResLayer):
            for layer in m.layers:
                if isinstance(layer, nn.Linear):
                    params.append(layer.weight)
                    if layer.bias is not None:
                        params.append(layer.bias)
    return coeff * sum((param**2).sum() for param in params)

class VAMPNet(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(VAMPNet, self).__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.layers = nn.Sequential(
            nn.Linear(inp_dim, width),
            ResLayer(width),
            ResLayer(width),
            ResLayer(width),
            nn.Linear(width, width),
            ResLayer(width),
            ResLayer(width),
            ResLayer(width),
            nn.Linear(width, out_dim)
        )
    def forward(self, x):
        return self.layers(x)

def make_VAMPNet(in_dim, out_dim=16, device="cuda"):
    model = VAMPNet(in_dim, out_dim).to(device)
    return model


def batched_model_eval(model, input, batch=16384):
    """ to avoid running out of memory, evaluate the model on a large tensor in batches
        Should only be called within torch.no_grad() context!
        model - the pytorch model to evaluate
        input - the input we are feeding to the model. shape: (N, channels)
        returns: the result of evaulating the model. shape: (N, out_chan) """
    N, channels = input.shape
    ans = torch.zeros(N, model.out_dim, device=input.device)
    for i in range(0, N, batch):
        ans[i:i+batch] = model(input[i:i+batch])
    return ans


class KoopmanModel:
    """ class containing a VAMPNet and the full set of associated transformations """
    def __init__(self, in_dim, out_dim, model, trans_0, trans_1, S):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.model = model
        self.model.eval() # put model into eval mode
        self.trans_0 = trans_0
        self.trans_1 = trans_1
        self.S = S
    @staticmethod
    def fromdata(model, dataset):
        N, L, in_dim = dataset.shape
        with torch.no_grad():
            chi = batched_model_eval(model, dataset.reshape(N*L, in_dim))
            chi_0 = chi.reshape(N, L, -1)[:, :-1].reshape(N*(L-1), -1)
            chi_1 = chi.reshape(N, L, -1)[:, 1: ].reshape(N*(L-1), -1)
            trans_0, trans_1, K = vamp_score(chi_0, chi_1, mode="all")
            trans_0, trans_1, K = trans_0.detach(), trans_1.detach(), K.detach()
        U, S, Vh = torch.linalg.svd(K)
        dim, = S.shape
        trans_0 = Affine( (Vh @ trans_0.W)[:dim], trans_0.mu[:dim])
        trans_1 = Affine((U.T @ trans_1.W)[:dim], trans_1.mu[:dim])
        return KoopmanModel(model.inp_dim, model.out_dim, model, trans_0, trans_1, S)
    @staticmethod
    def load(path):
        states = torch.load(path)
        model = VAMPNet(states["in_dim"], states["out_dim"]).to("cuda")
        model.load_state_dict(states["model"])
        return KoopmanModel(states["in_dim"], states["out_dim"], model,
            states["trans_0"], states["trans_1"], states["S"])
    def eigenfn_0(self, data):
        """ compute eigenfunction 0 on data """
        return self.trans_0(batched_model_eval(self.model, data))
    def eigenfn_1(self, data):
        """ compute eigenfunction 1 on data """
        return self.trans_1(batched_model_eval(self.model, data))
    def save(self, path):
        torch.save({
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "model": self.model.state_dict(),
            "trans_0": self.trans_0,
            "trans_1": self.trans_1,
            "S": self.S,
        }, path)
    def eval_score(self, dataset):
        """ evaluate performance of this model on a test dataset """
        N, L, _ = dataset.shape
        with torch.no_grad():
            x = self.eigenfn_0(dataset[:, :-1].reshape(N*(L-1), -1))
            y = self.eigenfn_1(dataset[:, 1: ].reshape(N*(L-1), -1))
            mu_x = x.mean(0)
            mu_y = y.mean(0)
            var_x = (x**2).mean(0) - mu_x**2
            var_y = (y**2).mean(0) - mu_y**2
            corr = (x*y).mean(0) - mu_x*mu_y
            ratio = corr / torch.sqrt(var_x*var_y)
        return ratio




