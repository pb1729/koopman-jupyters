import torch
import numpy as np
import matplotlib.pyplot as plt

from wgan import state_dim, GANTrainer, SubtractMeanPos
from sims import sims, get_dataset



# callables:
sub_mean_pos = SubtractMeanPos(state_dim) # module to allow subtracting mean position


def get_continuation_dataset(N, contins):
  print("creating initial states...")
  initial_states = get_dataset(sims["1D Polymer, Ornstein Uhlenbeck"], N, 1)[:, 0]
  print("created.")
  initial_states = sub_mean_pos(initial_states)
  # expand along another dim to enumerate continuations
  xv_init = initial_states[:, None].expand(-1, contins, -1).clone() # will be written to, so clone
  xv_init = xv_init.reshape(N*contins, state_dim)
  print("calculating continuations from initial states...")
  xv_fin = sims["1D Polymer, Ornstein Uhlenbeck"].generate_trajectory_xv(N*contins, 1, xv_init)
  print("done.")
  xv_fin = xv_fin.reshape(N, contins, state_dim)
  return initial_states, xv_fin


def get_sample_step(gan_trainer):
  """ given a GANTrainer class and current state, predict the next state """
  gan_trainer.set_eval(True)
  def sample_step(xv_init):
    batch, _ = xv_init.shape
    latents = gan_trainer.get_latents(batch)
    with torch.no_grad():
      xv_fin = gan_trainer.gen(latents, xv_init)
    return xv_fin + xv_init # gan predicts difference from previous step
  return sample_step


def compare_predictions(xv_predicted, xv_actual):
  xv_predicted = xv_predicted.cpu().numpy()
  xv_actual = xv_actual.cpu().numpy()
  fig = plt.figure(figsize=(20, 12))
  axes = []
  def rouse(n, j): # TODO: check that this is actually correct???
    return np.cos(n*np.pi*j/12) - 0.5*(n == 0)
  for i in range(12):
    w_x = rouse(i, np.arange(24))[None]
    w_x[:, 12:] = 0
    w_v = rouse(i, np.arange(24))[None]
    w_v[:, :12] = 0
    x_actual = (w_x*xv_actual).sum(1)
    #v_actual = (w_x*xv_actual).sum(1) # ignore velocity for now, because ornstein-uhlenbeck
    x_predicted = (w_x*xv_predicted).sum(1)
    axes.append(fig.add_subplot(3, 4, i + 1))
    axes[-1].hist(x_actual,    range=(-10., 10.), bins=100, alpha=0.7)
    axes[-1].hist(x_predicted, range=(-10., 10.), bins=100, alpha=0.7)
  plt.show()


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
    pred_fin_states = sample_step(init_state.expand(contins, state_dim))
    compare_predictions(pred_fin_states, fin_states)


def main(fpath):
  assert fpath is not None
  gan = GANTrainer.load(fpath)
  sample_step = get_sample_step(gan)
  init_states, fin_states = get_continuation_dataset(20, 10000)
  init_states, fin_states = init_states.to(torch.float32), fin_states.to(torch.float32)
  print(fin_states.shape, init_states.shape)
  eval_sample_step(sample_step, init_states, fin_states)




if __name__ == "__main__":
  from sys import argv
  main(*argv[1:])






