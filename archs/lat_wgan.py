import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config, Condition


# BASE SOURCE CODE FOR CONDITIONAL WGAN
#  * Uses CIFAR-10
#  * Conditions on average color of the image


lr_d = 0.0008   # learning rate for discriminator
lr_g = 0.0003   # learning rate for generator
beta_1 = 0.5    # Adam parameter
beta_2 = 0.99   # Adam parameter
k_L = 1.        # Lipschitz constant

nz = 100        # Size of z latent vector (i.e. size of generator input)
ngf = 128       # Size of feature maps in generator
ndf = 128       # Size of feature maps in discriminator


class EncCondition(nn.Module):
  def __init__(self, config, dim_out):
    super(EncCondition, self).__init__()
    self.layers = nn.Sequential(
      nn.Linear(config.cond_dim, dim_out)
    )
  def forward(self, x):
    return self.layers(x)


class Discriminator(nn.Module):
  def __init__(self, config):
    super(Discriminator, self).__init__()
    cond_dim = config.cond_dim
    self.layers_1 = nn.Sequential(
      nn.Linear(cond_dim, ndf),
      nn.LeakyReLU(0.2, inplace=True))
    self.cond_enc_1 = EncCondition(config, ndf)
    self.layers_2 = nn.Sequential(
      nn.Linear(ndf, ndf),
      nn.BatchNorm1d(ndf),
      nn.LeakyReLU(0.2, inplace=True))
    self.cond_enc_2 = EncCondition(config, ndf)
    self.layers_3 = nn.Sequential(
      nn.Linear(ndf, ndf),
      nn.BatchNorm1d(ndf),
      nn.LeakyReLU(0.2, inplace=True))
    self.cond_enc_3 = EncCondition(config, ndf)
    self.layers_4 = nn.Sequential(
      nn.Linear(ndf, ndf),
      nn.BatchNorm1d(ndf),
      nn.LeakyReLU(0.2, inplace=True))
    self.cond_enc_4 = EncCondition(config, ndf)
    self.layers_5 = nn.Sequential(
      nn.Linear(ndf, ndf),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(ndf, 1, bias=False),
    )
  def forward(self, inp, cond):
    y1 = self.layers_1(inp) + self.cond_enc_1(cond)
    y2 = self.layers_2(y1) + self.cond_enc_2(cond)
    y3 = self.layers_3(y2) + self.cond_enc_3(cond)
    y4 = self.layers_4(y3) + self.cond_enc_4(cond)
    return self.layers_5(y4)


class Generator(nn.Module):
  def __init__(self, config):
    super(Generator, self).__init__()
    cond_dim = config.cond_dim
    self.layers_1 = nn.Sequential(
      nn.Linear(nz, ngf),
      nn.BatchNorm1d(ngf),
      nn.LeakyReLU(0.2, inplace=True))
    self.cond_enc_1 = EncCondition(config, ngf)
    self.layers_2 = nn.Sequential(
      nn.Linear(ngf, ngf),
      nn.BatchNorm1d(ngf),
      nn.LeakyReLU(0.2, inplace=True))
    self.cond_enc_2 = EncCondition(config, ngf)
    self.lat_enc_2 = nn.Sequential(
      nn.Linear(nz, ngf),
      nn.BatchNorm1d(ngf),
      nn.LeakyReLU(0.2, inplace=True))
    self.layers_3 = nn.Sequential(
      nn.Linear(ngf, ngf),
      nn.BatchNorm1d(ngf),
      nn.LeakyReLU(0.2, inplace=True))
    self.cond_enc_3 = EncCondition(config, ngf)
    self.layers_4 = nn.Sequential(
      nn.Linear(ngf, state_dim, bias=False))
  def forward(self, input, cond):
    z1, z2 = input
    y1 = self.layers_1(z1) + self.cond_enc_1(cond)
    y2 = self.layers_2(y1) + self.cond_enc_2(cond) + self.lat_enc_2(z2)
    y3 = self.layers_3(y2) + self.cond_enc_3(cond)
    return cond + self.layers_4(y3)



# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)


class LatentGANTrainer:
  def __init__(self, disc, gen, config):
    self.disc = disc
    self.gen  = gen
    self.config = config
    self.init_optim()
  def init_optim(self):
    self.optim_d = torch.optim.Adam(self.disc.parameters(), lr_d, (beta_1, beta_2))
    self.optim_g = torch.optim.Adam(self.gen.parameters(),  lr_g, (beta_1, beta_2))
  @staticmethod
  def load_from_dict(states, config):
    disc, gen = Discriminator(config).to(config.device), Generator(config).to(config.device)
    disc.load_state_dict(states["disc"])
    gen.load_state_dict(states["gen"])
    return LatentGANTrainer(disc, gen, config)
  @staticmethod
  def makenew(config):
    disc, gen = Discriminator(config).to(config.device), Generator(config).to(config.device)
    disc.apply(weights_init)
    gen.apply(weights_init)
    return LatentGANTrainer(disc, gen, config)
  def save_to_dict(self):
    return {
        "disc": self.disc.state_dict(),
        "gen": self.gen.state_dict(),
      }
  def train_step(self, data, cond):
    loss_d = self.disc_step(data, cond)
    loss_g = self.gen_step(cond)
    return loss_d, loss_g
  def disc_step(self, r_data, cond):
    self.optim_d.zero_grad()
    # train on real data
    y_r = self.disc(r_data, cond)
    # train on generated data
    g_data = self.gen(self.get_latents(cond.shape[0]), cond)
    y_g = self.disc(g_data, cond)
    # sample-delta penalty on interpolated data
    mix_factors1 = torch.rand(cond.shape[0], 1, device=self.config.device)
    mixed_data1 = mix_factors1*g_data + (1 - mix_factors1)*r_data
    y_mixed1 = self.disc(mixed_data1, cond)
    mix_factors2 = torch.rand(cond.shape[0], 1, device=self.config.device)
    mixed_data2 = mix_factors2*g_data + (1 - mix_factors2)*r_data
    y_mixed2 = self.disc(mixed_data2, cond)
    ep_penalty = (self.endpoint_penalty(r_data, g_data, y_r, y_g)
                + self.endpoint_penalty(mixed_data1, r_data, y_mixed1, y_r)
                + self.endpoint_penalty(g_data, mixed_data1, y_g, y_mixed1)
                + self.endpoint_penalty(mixed_data2, r_data, y_mixed2, y_r)
                + self.endpoint_penalty(g_data, mixed_data2, y_g, y_mixed2)
                + self.endpoint_penalty(mixed_data1, mixed_data2, y_mixed1, y_mixed2))
    # loss, backprop, update
    loss = ep_penalty.mean() + y_r.mean() - y_g.mean()
    loss.backward()
    self.optim_d.step()
    return loss.item()
  def gen_step(self, cond):
    self.optim_g.zero_grad()
    g_data = self.gen(self.get_latents(cond.shape[0]), cond)
    y_g = self.disc(g_data, cond)
    loss = y_g.mean()
    loss.backward()
    self.optim_g.step()
    return loss.item()
  def endpoint_penalty(self, x1, x2, y1, y2):
    dist = torch.sqrt(((x1 - x2)**2).mean(1))
    # one-sided L1 penalty:
    penalty = F.relu(torch.abs(y1 - y2)/(dist*k_L + 1e-6) - 1.)
    # weight by square root of separation
    return torch.sqrt(dist)*penalty
  def get_latents(self, batchsz):
    """ sample latents for generator """
    return (
      torch.randn(batchsz, nz, device=self.config.device),
      torch.randn(batchsz, nz, device=self.config.device))
  def set_eval(self, bool_eval):
    if bool_eval:
      self.disc.eval()
      self.gen.eval()
    else:
      self.disc.train()
      self.gen.train()



# export model class:
modelclass = LatentGANTrainer



