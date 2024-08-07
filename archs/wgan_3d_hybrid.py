import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

from config import Condition
from utils import must_be
from gan_common import GANTrainer
from layers_common import *


# constants:
k_L = 1.       # Lipschitz constant


class LocalResidual(nn.Module):
  """ Residual layer that just does local processing on individual nodes. """
  def __init__(self, config):
    super().__init__()
    adim, vdim, rank = config["adim"], config["vdim"], config["rank"]
    agroups, vgroups = config["agroups"], config["vgroups"]
    self.layers_a = nn.Sequential(
      nn.Linear(adim, adim),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(adim, adim),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(adim, adim))
    self.layers_v = nn.Sequential(
      VecLinear(vdim, vdim),
      VecRootS(),
      VecLinear(vdim, vdim),
      VecRootS(),
      VecLinear(vdim, vdim))
    self.av_prod = ScalVecProducts(adim, vdim, rank)
    self.gnorm_a = ScalGroupNorm(adim, agroups)
    self.gnorm_v = VecGroupNorm(vdim, vgroups)
  def forward(self, x_a, x_v):
    y_a = self.layers_a(x_a)
    y_v = self.layers_v(x_v)
    p_a, p_v = self.av_prod(x_a, x_v)
    z_a, z_v = self.gnorm_a(y_a + p_a), self.gnorm_v(y_v + p_v)
    return (x_a + z_a), (x_v + z_v)


class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    adim, vdim = config["adim"], config["vdim"]
    agroups, vgroups = config["agroups"], config["vgroups"]
    self.edge_embed = EdgeRelativeEmbedMLPPath(adim, vdim)
    self.node_embed = NodeRelativeEmbedMLP(adim, vdim)
    self.conv_0_a = ScalConv1d(adim, 7)
    self.conv_0_v = VecConv1d(vdim, 7)
    self.local_res = LocalResidual(config)
    self.conv_1_a = ScalConv1d(adim, 7)
    self.conv_1_v = VecConv1d(vdim, 7)
    self.gnorm_a = ScalGroupNorm(adim, agroups)
    self.gnorm_v = VecGroupNorm(vdim, vgroups)
  def get_embedding(self, pos_0, pos_1):
    emb_edge_a, emb_edge_v = self.edge_embed(pos_0, pos_1)
    emb_node_a, emb_node_v = self.node_embed(pos_0, pos_1)
    return emb_edge_a + emb_node_a, emb_edge_v + emb_node_v
  def forward(self, tup):
    pos_0, pos_1, x_a, x_v = tup
    emb_a, emb_v = self.get_embedding(pos_0, pos_1)
    y_a = self.conv_0_a(x_a) + emb_a
    y_v = self.conv_0_v(x_v) + emb_v
    y_a, y_v = self.local_res(y_a, y_v)
    z_a = self.gnorm_a(self.conv_1_a(y_a))
    z_v = self.gnorm_v(self.conv_1_v(y_v))
    return pos_0, pos_1, (x_a + z_a), (x_v + z_v)



class Discriminator(nn.Module):
  def __init__(self, config):
    super().__init__()
    adim, vdim = config["adim"], config["vdim"]
    self.node_enc = NodeRelativeEmbedMLP(adim, vdim)
    self.blocks = nn.Sequential(
      Block(config),
      Block(config),
      Block(config),
      Block(config))
    self.lin_a = nn.Linear(adim, 1, bias=False)
    self.lin_v = nn.Linear(vdim, 1, bias=False)
  def forward(self, pos_0, pos_1):
    x_a, x_v = self.node_enc(pos_0, pos_1)
    *_, y_a, y_v = self.blocks((pos_0, pos_1, x_a, x_v))
    v_norms = torch.linalg.vector_norm(y_v, dim=-1)
    return (self.lin_a(y_a) + self.lin_v(v_norms)).sum(2).mean(1)


class Generator(nn.Module):
  def __init__(self, config):
    super().__init__()
    adim, vdim = config["adim"], config["vdim"]
    self.node_enc = NodeRelativeEmbedMLP(adim, vdim)
    self.blocks = nn.Sequential(
      Block(config),
      Block(config),
      Block(config))
    self.blocks_tune = nn.Sequential(
      Block(config),
      Block(config),
      Block(config))
    # some wastage of compute here, since the final sets of scalar values are not really used
    self.lin_v_node = VecLinear(vdim, 1)
    self.lin_v_node_tune = VecLinear(vdim, 1)
    self.out_norm_coeff = vdim**(-0.5)
  def _predict(self, pos0, noised):
    x_a, x_v = self.node_enc(pos0, noised)
    *_, y_v = self.blocks((pos0, noised, x_a, x_v))
    return self.lin_v_node(y_v)[:, :, 0]*self.out_norm_coeff
  def _finetune(self, pos0, noised, ε_a, ε_v):
    x_a, x_v = self.node_enc(pos0, noised)
    x_a = x_a + ε_a
    x_v = x_v + ε_v
    *_, y_v = self.blocks_tune((pos0, noised, x_a, x_v))
    return self.lin_v_node_tune(y_v)[:, :, 0]*self.out_norm_coeff
  def forward(self, pos0, noise, ε_a, ε_v):
    noised = pos0 + noise
    pred_noise = self._predict(pos0, noised)
    noised = noised - pred_noise
    return noised + self._finetune(pos0, noised, ε_a, ε_v)



class WGAN3D:
  is_gan = True
  def __init__(self, crow, taxi, gen, config):
    self.crow = crow # discriminator
    self.taxi = taxi # discriminator
    self.gen  = gen
    self.config = config
    assert config.sim.space_dim == 3
    assert config.cond_type == Condition.COORDS
    assert config.subtract_mean == 0
    assert config.x_only
    self.n_nodes = config.sim.poly_len
    self.init_optim()
  def init_optim(self):
    betas = (self.config["beta_1"], self.config["beta_2"])
    self.optim_d = torch.optim.AdamW(chain(self.crow.parameters(), self.taxi.parameters()),
      self.config["lr_d"], betas, weight_decay=self.config["weight_decay"])
    self.optim_g = torch.optim.AdamW(self.gen.parameters(), 
      self.config["lr_g"], betas, weight_decay=self.config["weight_decay"])
  @staticmethod
  def load_from_dict(states, config):
    crow, taxi, gen = Discriminator(config).to(config.device), Discriminator(config).to(config.device), Generator(config).to(config.device)
    crow.load_state_dict(states["crow"])
    taxi.load_state_dict(states["taxi"])
    gen.load_state_dict(states["gen"])
    return WGAN3D(crow, taxi, gen, config)
  @staticmethod
  def makenew(config):
    crow, taxi, gen = Discriminator(config).to(config.device), Discriminator(config).to(config.device), Generator(config).to(config.device)
    crow.apply(weights_init)
    taxi.apply(weights_init)
    gen.apply(weights_init)
    return WGAN3D(crow, taxi, gen, config)
  def save_to_dict(self):
    return {
        "crow": self.crow.state_dict(),
        "taxi": self.taxi.state_dict(),
        "gen": self.gen.state_dict(),
      }
  def train_step(self, data, cond):
    data, cond = data.reshape(-1, self.n_nodes, 3), cond.reshape(-1, self.n_nodes, 3)
    loss_d = self.disc_step(data, cond)
    loss_g = self.gen_step(cond)
    return loss_d, loss_g
  def disc_step(self, r_data, cond):
    self.optim_d.zero_grad()
    # train on real data
    y_r_crow, y_r_taxi = self.y_disc(cond, r_data)
    # train on generated data
    g_data = self.generate(cond)
    y_g_crow, y_g_taxi = self.y_disc(cond, g_data)
    # endpoint penalty on interpolated data, each discriminator gets its own penalty
    mix_factors = torch.rand(cond.shape[0], 1, 1, device=self.config.device)
    m_data = mix_factors*g_data + (1 - mix_factors)*r_data
    y_m_crow, y_m_taxi = self.y_disc(cond, m_data)
    penalty_crow = (self.endpoint_penalty_crow(r_data, g_data, y_r_crow, y_g_crow)
                  + self.endpoint_penalty_crow(r_data, m_data, y_r_crow, y_m_crow)
                  + self.endpoint_penalty_crow(m_data, g_data, y_m_crow, y_g_crow))
    penalty_taxi = (self.endpoint_penalty_taxi(r_data, g_data, y_r_taxi, y_g_taxi)
                  + self.endpoint_penalty_taxi(r_data, m_data, y_r_taxi, y_m_taxi)
                  + self.endpoint_penalty_taxi(m_data, g_data, y_m_taxi, y_g_taxi))
    # loss, backprop, update
    loss = (y_r_crow.mean() + y_r_taxi.mean()) - (y_g_crow.mean() + y_g_taxi.mean())
    loss = loss + self.config["lpen_wt"]*(penalty_crow.mean() + penalty_taxi.mean())
    loss.backward()
    self.optim_d.step()
    return loss.item()
  def gen_step(self, cond):
    self.optim_g.zero_grad()
    g_data = self.generate(cond)
    y_g_crow, y_g_taxi = self.y_disc(cond, g_data)
    y_g = y_g_crow + y_g_taxi
    loss = y_g.mean()
    loss.backward()
    self.optim_g.step()
    return loss.item()
  def endpoint_penalty_crow(self, x1, x2, y1, y2):
    # use the euclidean metric (take RMS distance)
    dist = torch.sqrt(((x1 - x2)**2).sum(2).mean(1))
    # one-sided L1 penalty:
    penalty = F.relu(torch.abs(y1 - y2)/(dist*k_L + 1e-6) - 1.)
    # weight by square root of separation
    return torch.sqrt(dist)*penalty
  def endpoint_penalty_taxi(self, x1, x2, y1, y2):
    # use the taxicab metric (take average distance that node moved rather than RMS distance)
    dist = torch.sqrt(((x1 - x2)**2).sum(2)).mean(1)
    # one-sided L1 penalty:
    penalty = F.relu(torch.abs(y1 - y2)/(dist*k_L + 1e-6) - 1.)
    # weight by square root of separation
    return torch.sqrt(dist)*penalty
  def generate(self, cond):
    batch, must_be[self.n_nodes], must_be[3] = cond.shape
    pos_noise, z_a, z_v = self.get_latents(batch)
    return self.gen(cond, pos_noise, z_a, z_v)
  def y_disc(self, x0, x1):
    y_crow = self.crow(x0, x1)
    y_taxi = self.taxi(x0, x1)
    return y_crow, y_taxi
  def get_latents(self, batchsz):
    """ sample latent space for generator """
    pos_noise = self.config["z_scale"]*torch.randn(batchsz, self.n_nodes, 3, device=self.config.device)
    z_a = torch.randn(batchsz, self.n_nodes, self.config["adim"], device=self.config.device)
    z_v = torch.randn(batchsz, self.n_nodes, self.config["vdim"], 3, device=self.config.device)
    return pos_noise, z_a, z_v
  def set_eval(self, bool_eval):
    if bool_eval:
      self.crow.eval()
      self.taxi.eval()
      self.gen.eval()
    else:
      self.crow.train()
      self.taxi.train()
      self.gen.train()
  def predict(self, cond):
    batch, must_be[3*self.n_nodes] = cond.shape
    with torch.no_grad():
      return self.generate(cond.reshape(batch, self.n_nodes, 3)).reshape(batch, 3*self.n_nodes)



# export model class and trainer class:
modelclass   = WGAN3D
trainerclass = GANTrainer



