import torch
import importlib

from sims import sims
from vampnets import KoopmanModel
from polymer_util import rouse_block


ARCH_PREFIX = "archs."


class Condition:
  COORDS = 1
  ROUSE = 2
  KOOPMODEL = 3



class Config:
  """ configuration class for training runs """
  def __init__(self, sim_name, arch_name,
               cond=Condition.COORDS, x_only=False, subtract_mean=0, device="cuda",
               batch=16, simlen=16,
               koopman_model_path=None, n_rouse_modes=None):
    self.sim_name = sim_name
    self.arch_name = arch_name
    self.sim = sims[self.sim_name]
    self.cond_type = cond
    self.x_only = x_only
    self.subtract_mean = subtract_mean # 0 if we don't subtract mean, otherwise is the number of dimensions that an individual particle moves in
    self.device = device
    self.batch = batch
    self.simlen = simlen
    if self.x_only:
      self.state_dim = self.sim.dim
    else:
      self.state_dim = 2*self.sim.dim
    if cond == Condition.COORDS:
      self.cond_coords()
    elif cond == Condition.ROUSE:
      assert n_rouse_modes is not None, "must specify a number of Rouse modes to use"
      self.cond_rouse(n_rouse_modes)
    elif cond == Condition.KOOPMODEL:
      assert koopman_model_path is not None, "must specify a path to obtain the Koopman model from"
      self.cond_koopman(koopman_model_path)
    else:
      assert False, "condition not recognized!"
    self.modelclass = self.get_modelclass()
  def cond_coords(self):
    """ given a state dim, the condition is given by the entire state """
    self.cond = (lambda state: state)
    self.cond_dim = self.state_dim
  def cond_koopman(self, model_path):
    """ given a KoopmanModel, the condition is given by that model """
    assert not self.subtract_mean, "should not subtract mean if we're using the Koopman eigenfunctions"
    self.koopman_model_path = model_path
    kmod = KoopmanModel.load(model_path)
    def get_cond_koopman(data):
      with torch.no_grad():
        ans = kmod.eigenfn_0(data)
      return ans.detach()
    self.cond = get_cond_koopman
    self.cond_dim = kmod.out_dim
  def cond_rouse(self, n_modes):
    assert self.x_only, "Rouse modes with v not implemented! x_only should be True"
    assert hasattr(self.sim, "poly_len"), "simulation should specify a polymer length"
    self.n_rouse_modes = n_modes
    blk = torch.tensor(rouse_block(self.sim.poly_len)[:, :n_modes], device=self.device, dtype=torch.float32)
    if self.subtract_mean:
      blk = blk[:, 1:] # if we're subtracting the mean, 0th mode is meaningless
    def get_cond_rouse(data):
      """ condition on n_modes Rouse modes """
      return data @ blk # (batch, poly_len) @ (poly_len, n_modes)
    self.cond = get_cond_rouse
    self.cond_dim = blk.shape[1] # due to subtract_mean, can *not* use n_modes here
  def get_modelclass(self):
    arch_module = importlib.import_module(ARCH_PREFIX + self.arch_name)
    return arch_module.modelclass
  def get_args_and_kwargs(self):
    """ find the args and kwargs that one would need to reconstruct this config """
    args = [self.sim_name, self.arch_name]
    kwargs = {
      "cond": self.cond_type,
      "x_only": self.x_only,
      "subtract_mean": self.subtract_mean,
      "device": self.device,
      "batch": self.batch,
      "simlen": self.simlen,
      }
    for attr in ["koopman_model_path", "n_rouse_modes"]:
      if hasattr(self, attr):
        kwargs[attr] = getattr(self, attr)
    return args, kwargs



# universal functions for models. models must have the following behaviours:
# * .config                         is the config for that model
# * @staticmethod .load_from_dict() creates the model from a dict of states
# * .save_to_dict()                 records the model states into a dictionary
# * @staticmethod .makenew()        creates a new instance of the model

def load(path):
  data = torch.load(path)
  config = Config(*data["args"], **data["kwargs"])
  return config.modelclass.load_from_dict(data["states"], config)

def save(model, path):
  config_args, config_kwargs = model.config.get_args_and_kwargs()
  states = model.save_to_dict()
  torch.save({
      "args": config_args,
      "kwargs": config_kwargs,
      "states": states,
    }, path)

def makenew(config):
  return config.modelclass.makenew(config)




