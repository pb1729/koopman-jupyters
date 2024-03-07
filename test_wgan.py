import torch
import numpy as np
import matplotlib.pyplot as plt

from wgan import GANTrainer
from train_wgan import GET_COND, COND_DIM
from sims import sims, get_dataset
from polymer_util import rouse



def get_continuation_dataset(N, contins, subtract_cm=0, x_only=False):
  """ get a dataset of many possible continued trajectories from each of N initial states """
  print("creating initial states...") # equilibriate for 120 steps...
  initial_states = get_dataset(sims["1D Polymer, Ornstein Uhlenbeck"], N, 1, t_eql=120, subtract_cm=subtract_cm)[:, 0]
  print("created.")
  # expand along another dim to enumerate continuations
  state_dim = initial_states.shape[-1]
  xv_init = initial_states[:, None].expand(N, contins, state_dim).clone() # will be written to, so clone
  xv_init = xv_init.reshape(N*contins, state_dim)
  print("calculating continuations from initial states...")
  xv_fin = get_dataset(sims["1D Polymer, Ornstein Uhlenbeck"], N*contins, 1, subtract_cm=subtract_cm,
    x_init=xv_init[:, :state_dim//2], v_init=xv_init[:, state_dim//2:])
  print("done.")
  xv_fin = xv_fin.reshape(N, contins, state_dim)
  if x_only:
    return initial_states[:, :state_dim//2], xv_fin[:, :, :state_dim//2]
  return initial_states, xv_fin


def get_sample_step(gan_trainer):
  """ given a GANTrainer class and current state, predict the next state """
  gan_trainer.set_eval(True)
  def sample_step(xv_init):
    batch, _ = xv_init.shape
    latents = gan_trainer.get_latents(batch)
    with torch.no_grad():
      xv_fin = gan_trainer.gen(latents, GET_COND(xv_init))
    return xv_fin # TODO: see related note in train_wgan #+ xv_init # gan predicts difference from previous step
  return sample_step


def compare_predictions_x(x_predicted, x_actual):
  x_actual = x_actual.cpu().numpy()
  x_predicted = x_predicted.cpu().numpy()
  fig = plt.figure(figsize=(20, 12))
  axes = []
  for n in range(12): # TODO: leave polymer length as a variable???
    w_x = rouse(n, 12)[None]
    x_actual_n = (w_x*x_actual).sum(1)
    x_predicted_n = (w_x*x_predicted).sum(1)
    axes.append(fig.add_subplot(3, 4, n + 1))
    axes[-1].hist(x_actual_n,    range=(-20., 20.), bins=200, alpha=0.7)
    axes[-1].hist(x_predicted_n, range=(-20., 20.), bins=200, alpha=0.7)
  #plt.show()
  plt.savefig("figures/pos/%d.png" % np.random.randint(0, 10000))
  plt.close()


def eval_sample_step(sample_step, init_states, fin_statess):
  """ given a method that continues evolution for one more step,
      plot various graphs to evaluate it for accuracy
      sample_step: (batch, state_dim) -> (batch, state_dim)
      init_states: (batch, state_dim)
      fin_statess: (batch, contins, state_dim) """
  batch, contins, _ = fin_statess.shape
  for i in range(batch):
    init_state = init_states[i, None]
    fin_states = fin_statess[i]
    pred_fin_states = sample_step(init_state.expand(contins, -1))
    compare_predictions_x(pred_fin_states, fin_states)


def main(fpath):
  assert fpath is not None
  gan = GANTrainer.load(fpath)
  gan.set_eval(True)
  sample_step = get_sample_step(gan)
  init_states, fin_states = get_continuation_dataset(20, 10000, subtract_cm=1, x_only=True)
  init_states, fin_states = init_states.to(torch.float32), fin_states.to(torch.float32)
  print(fin_states.shape, init_states.shape)
  eval_sample_step(sample_step, init_states, fin_states)




if __name__ == "__main__":
  from sys import argv
  main(*argv[1:])






