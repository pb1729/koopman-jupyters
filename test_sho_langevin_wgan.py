import torch
import numpy as np
import matplotlib.pyplot as plt

from wgan import GANTrainer, GET_COND, COND_DIM, SIM, SUBTRACT_MEAN, X_ONLY
from sims import sims, get_dataset



def get_continuation_dataset(N, contins, subtract_cm=0, x_only=False):
  """ get a dataset of many possible continued trajectories from each of N initial states """
  print("creating initial states...") # equilibriate for 120 steps...
  initial_states = get_dataset(SIM, N, 1, t_eql=120, subtract_cm=SUBTRACT_MEAN)[:, 0]
  print("created.")
  # expand along another dim to enumerate continuations
  state_dim = initial_states.shape[-1]
  xv_init = initial_states[:, None].expand(N, contins, state_dim).clone() # will be written to, so clone
  xv_init = xv_init.reshape(N*contins, state_dim)
  print("calculating continuations from initial states...")
  xv_fin = get_dataset(SIM, N*contins, 1, subtract_cm=SUBTRACT_MEAN,
    x_init=xv_init[:, :state_dim//2], v_init=xv_init[:, state_dim//2:])
  print("done.")
  xv_fin = xv_fin.reshape(N, contins, state_dim)
  if X_ONLY:
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


def compare_predictions_x(x_predicted, x_actual, init_state=None):
  x_actual = x_actual.cpu().numpy()
  x_predicted = x_predicted.cpu().numpy()
  fig = plt.figure(figsize=(20, 12))
  axes = []
  axes.append(fig.add_subplot(1, 2, 1))
  axes[-1].hist2d(x_actual[:, 0], x_actual[:, 1],
    bins=40, range=[[-2.5, 2.5], [-2.5, 2.5]])
  axes.append(fig.add_subplot(1, 2, 2))
  axes[-1].hist2d(x_predicted[:, 0], x_predicted[:, 1],
    bins=40, range=[[-2.5, 2.5], [-2.5, 2.5]])
  for ax in axes:
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    if init_state is not None:
      print(init_state.shape)
      print(init_state)
      ax.plot(init_state[:, 0].cpu(), init_state[:, 1].cpu(), color="red", marker="o")
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
    pred_fin_states = sample_step(init_state.expand(contins, -1))
    compare_predictions_x(pred_fin_states, fin_states, init_state=init_state)


def main(fpath):
  assert fpath is not None
  gan = GANTrainer.load(fpath)
  gan.set_eval(True)
  sample_step = get_sample_step(gan)
  init_states, fin_states = get_continuation_dataset(20, 10000, subtract_cm=1, x_only=True)
  init_states, fin_states = init_states.to(torch.float32), fin_states.to(torch.float32)
  print(fin_states.shape, init_states.shape)
  eval_sample_step(sample_step, init_states, fin_states)


def main_instability_demo():
  def demo(dt):
    x = [2.]
    v = [0.]
    for t in range(10):
      x.append(x[-1] + dt*v[-1])
      v.append(v[-1] - dt*x[-1])
    print(len(x), len(v), x, v)
    plt.plot(x, v, marker="o")
    plt.gca().set_aspect("equal")
    plt.title("dt = %f" % dt)
    plt.xlabel("x")
    plt.ylabel("p")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.show()
  demo(0.1)
  demo(0.8)
  demo(2.1)



if __name__ == "__main__":
  if True:
    from sys import argv
    main(*argv[1:])
  else:
    main_instability_demo()





