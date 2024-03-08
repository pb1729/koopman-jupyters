import torch
import torch.nn as nn
import torch.nn.functional as F

from sims import sims


# BASE SOURCE CODE FOR CONDITIONAL WGAN
#  * Uses CIFAR-10
#  * Conditions on average color of the image


device = "cuda"
batch = 32      # batch size
lr_d = 0.0008   # learning rate for discriminator
lr_g = 0.0003   # learning rate for generator
beta_1 = 0.5    # Adam parameter
beta_2 = 0.99   # Adam parameter
k_L = 1.        # Lipschitz constant
in_str = 0.2    # Strength of Instance-Noise

nz = 100        # Size of z latent vector (i.e. size of generator input)
ngf = 64        # Size of feature maps in generator
ndf = 64        # Size of feature maps in discriminator


# TODO: maybe make it easier to change conditioning???

def get_get_cond_all(state_dim):
  """ given a state dim, the condition is given by the entire state """
  def get_cond_all(data):
    return data
  return get_cond_all, state_dim


def get_get_cond_koopman(koopman_model):
  """ given a KoopmanModel, the condition is given by that model """
  def get_cond_koopman(data):
    with torch.no_grad():
      ans = koopman_model.eigenfn_0(data)
    return ans.detach()
  return get_cond_koopman, koopman_model.out_dim


# define conditioning for this run:
X_ONLY = True # True if coordinates are (x) and False if coordinates are (x,v)
USE_KOOPMAN_EIGENFN = False
#SIM = sims["SHO, Langevin"]
SIM = sims["1D Polymer, Ornstein Uhlenbeck"]
if X_ONLY:
  STATE_DIM = SIM.dim
else:
  STATE_DIM = 2*SIM.dim
if USE_KOOPMAN_EIGENFN:
  GET_COND, COND_DIM = get_get_cond_koopman(KoopmanModel.load("models/e_1.koop.pt"))
else:
  GET_COND, COND_DIM = get_get_cond_all(STATE_DIM)
SUBTRACT_MEAN = 0 # 0 if we don't subtract mean, otherwise is the number of dimensions that an individual particle moves in
assert not (SUBTRACT_MEAN and USE_KOOPMAN_EIGENFN) # should not subtract mean if we're using the Koopman eigenfunctions...


def get_input_preproc(state_dim):
  """ preprocess input by subtracting mean if necessary """
  layer_list = []
  if SUBTRACT_MEAN:
    layer_list.append(SubtractMean(state_dim, SUBTRACT_MEAN) if X_ONLY else SubtractMeanPos(state_dim, SUBTRACT_MEAN))
  return nn.Sequential(*layer_list)


class EncCondition(nn.Module):
  def __init__(self, state_dim, dim_out, space_dim=1):
    super(EncCondition, self).__init__()
    self.layers = nn.Sequential(
      get_input_preproc(state_dim),
      nn.Linear(state_dim, dim_out)
    )
  def forward(self, x):
    return self.layers(x)


class SubtractMean(nn.Module):
  """ given position coordinates as input, subtracts the mean """
  def __init__(self, state_dim, space_dim=1):
    super(SubtractMean, self).__init__()
    self.x_sz = state_dim
    self.space_dim = space_dim # number of spatial dimensions
    assert self.x_sz % self.space_dim == 0
  def forward(self, x):
    batch, state_dim = x.shape
    x = x.reshape(batch, -1, self.space_dim)
    x = x - x.mean(1, keepdim=True)
    return x.reshape(batch, self.x_sz)


class SubtractMeanPos(nn.Module):
  """ given position and velocity coordinates as input,
      subtracts the mean from the position coordinates only """
  def __init__(self, state_dim, space_dim=1):
    super(SubtractMeanPos, self).__init__()
    self.x_sz = state_dim // 2
    self.space_dim = space_dim # number of spatial dimensions
    assert self.x_sz % self.space_dim == 0
  def forward(self, xv):
    batch, state_dim = xv.shape
    x = xv[:, :self.x_sz].reshape(batch, -1, self.space_dim)
    x = x - x.mean(1, keepdim=True)
    return torch.cat([
      x.reshape(batch, self.x_sz),
      xv[:, self.x_sz:]], dim=1)



class Discriminator(nn.Module):
  def __init__(self, state_dim, cond_dim):
    super(Discriminator, self).__init__()
    self.layers_1 = nn.Sequential(
      get_input_preproc(state_dim),
      nn.Linear(state_dim, ndf),
      nn.LeakyReLU(0.2, inplace=True))
    self.cond_enc_1 = EncCondition(cond_dim, ndf)
    self.layers_2 = nn.Sequential(
      nn.Linear(ndf, ndf),
      nn.BatchNorm1d(ndf),
      nn.LeakyReLU(0.2, inplace=True))
    self.cond_enc_2 = EncCondition(cond_dim, ndf)
    self.layers_3 = nn.Sequential(
      nn.Linear(ndf, ndf),
      nn.BatchNorm1d(ndf),
      nn.LeakyReLU(0.2, inplace=True))
    self.cond_enc_3 = EncCondition(cond_dim, ndf)
    self.layers_4 = nn.Sequential(
      nn.Linear(ndf, ndf),
      nn.BatchNorm1d(ndf),
      nn.LeakyReLU(0.2, inplace=True))
    self.cond_enc_4 = EncCondition(cond_dim, ndf)
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
  def __init__(self, state_dim, cond_dim):
    super(Generator, self).__init__()
    self.layers_1 = nn.Sequential(
      nn.Linear(nz, ngf),
      nn.BatchNorm1d(ngf),
      nn.LeakyReLU(0.2, inplace=True))
    self.cond_enc_1 = EncCondition(cond_dim, ngf)
    self.layers_2 = nn.Sequential(
      nn.Linear(ngf, ngf),
      nn.BatchNorm1d(ngf),
      nn.LeakyReLU(0.2, inplace=True))
    self.cond_enc_2 = EncCondition(cond_dim, ngf)
    self.layers_3 = nn.Sequential(
      nn.Linear(ngf, ngf),
      nn.BatchNorm1d(ngf),
      nn.LeakyReLU(0.2, inplace=True))
    self.cond_enc_3 = EncCondition(cond_dim, ngf)
    self.layers_4 = nn.Sequential(
      nn.Linear(ngf, state_dim, bias=False))
  def forward(self, input, cond):
    y1 = self.layers_1(input) + self.cond_enc_1(cond)
    y2 = self.layers_2(y1) + self.cond_enc_2(cond)
    y3 = self.layers_3(y2) + self.cond_enc_3(cond)
    return self.layers_4(y3)


# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)


class GANTrainer:
  def __init__(self, disc, gen, state_dim, cond_dim):
    self.disc = disc
    self.gen  = gen
    self.state_dim = state_dim
    self.cond_dim = cond_dim
    self.init_optim()
  @staticmethod
  def load(path):
    states = torch.load(path)
    state_dim = states["state_dim"]
    cond_dim = states["cond_dim"]
    disc, gen = Discriminator(state_dim, cond_dim).to(device), Generator(state_dim, cond_dim).to(device)
    disc.load_state_dict(states["disc"])
    gen.load_state_dict(states["gen"])
    return GANTrainer(disc, gen, state_dim, cond_dim)
  @staticmethod
  def makenew(state_dim, cond_dim):
    disc, gen = Discriminator(state_dim, cond_dim).to(device), Generator(state_dim, cond_dim).to(device)
    disc.apply(weights_init)
    gen.apply(weights_init)
    return GANTrainer(disc, gen, state_dim, cond_dim)
  def init_optim(self):
    self.optim_d = torch.optim.Adam(self.disc.parameters(), lr_d, (beta_1, beta_2))
    self.optim_g = torch.optim.Adam(self.gen.parameters(),  lr_g, (beta_1, beta_2))
  def save(self, path):
    torch.save({
        "disc": self.disc.state_dict(),
        "gen": self.gen.state_dict(),
        "state_dim": self.state_dim,
        "cond_dim": self.cond_dim,
      }, path)
  def train_step(self, data, cond):
    loss_d = self.disc_step(data, cond)
    loss_g = self.gen_step(cond)
    return loss_d, loss_g
  def disc_step(self, data, cond):
    self.optim_d.zero_grad()
    # train on real data (with instance noise)
    r_data = data + in_str*torch.randn_like(data) # instance noise
    y_r = self.disc(r_data, cond)
    # train on generated data (with instance noise)
    g_data = self.gen(self.get_latents(cond.shape[0]), cond)
    g_data = g_data + in_str*torch.randn_like(g_data) # instance noise
    y_g = self.disc(g_data, cond)
    # sample-delta penalty on interpolated data
    mix_factors = torch.rand(cond.shape[0], 1, device=device)
    mixed_data = mix_factors*g_data + (1 - mix_factors)*r_data
    y_mixed = self.disc(mixed_data, cond)
    ep_penalty = (self.endpoint_penalty(r_data, g_data, y_r, y_g)
                + self.endpoint_penalty(mixed_data, r_data, y_mixed, y_r)
                + self.endpoint_penalty(g_data, mixed_data, y_g, y_mixed))
    # loss, backprop, update
    loss = ep_penalty.mean() + y_r.mean() - y_g.mean()
    loss.backward()
    self.optim_d.step()
    return loss.item()
  def gen_step(self, cond):
    self.optim_g.zero_grad()
    g_data = self.gen(self.get_latents(cond.shape[0]), cond)
    g_data = g_data + in_str*torch.randn_like(g_data) # instance noise
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
    return penalty*torch.sqrt(dist)
  def get_latents(self, batchsz):
    """ sample latents for generator """
    return torch.randn(batchsz, nz, device=device)
  def set_eval(self, bool_eval):
    if bool_eval:
      self.disc.eval()
      self.gen.eval()
    else:
      self.disc.train()
      self.gen.train()



