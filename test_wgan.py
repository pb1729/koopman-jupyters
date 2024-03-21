import torch
import numpy as np
import matplotlib.pyplot as plt

from configs import load
from sims import sims, get_dataset
from polymer_util import rouse


ITERATIONS = 1



def get_continuation_dataset(N, contins, config, iterations=1):
  """ get a dataset of many possible continued trajectories from each of N initial states """
  print("creating initial states...")
  initial_states = get_dataset(config.sim, N, 1, t_eql=config.t_eql, subtract_cm=config.subtract_mean)[:, 0]
  print("created.")
  # expand along another dim to enumerate continuations
  state_dim = initial_states.shape[-1] # this should include velocity, so we can't use config.state_dim, which may be x only
  xv_init = initial_states[:, None].expand(N, contins, state_dim).clone() # will be written to, so clone
  xv_init = xv_init.reshape(N*contins, state_dim)
  print("calculating continuations from initial states...")
  xv_fin = get_dataset(config.sim, N*contins, iterations, subtract_cm=config.subtract_mean,
    x_init=xv_init[:, :state_dim//2], v_init=xv_init[:, state_dim//2:])[:, -1]
  print("done.")
  xv_fin = xv_fin.reshape(N, contins, state_dim)
  if config.x_only:
    return initial_states[:, :state_dim//2], xv_fin[:, :, :state_dim//2]
  return initial_states, xv_fin


def get_sample_step(model):
  """ given a model and current state, predict the next state """
  model.set_eval(True)
  def sample_step(state):
    batch, _ = state.shape
    latents = model.get_latents(batch)
    with torch.no_grad():
      state_fin = model.gen(latents, model.config.cond(state))
    return state_fin
  return sample_step


def compare_predictions_x(x_init, x_predicted, x_actual):
  x_init = x_init.cpu().numpy()
  x_actual = x_actual.cpu().numpy()
  x_predicted = x_predicted.cpu().numpy()
  fig = plt.figure(figsize=(20, 12))
  axes = []
  for n in range(12): # TODO: leave polymer length as a variable???
    w_x = rouse(n, 12)[None]
    x_init_n = (w_x*x_init).sum(1)
    x_actual_n = (w_x*x_actual).sum(1)
    x_predicted_n = (w_x*x_predicted).sum(1)
    axes.append(fig.add_subplot(3, 4, n + 1))
    axes[-1].hist(x_actual_n,    range=(-20., 20.), bins=200, alpha=0.7)
    axes[-1].hist(x_predicted_n, range=(-20., 20.), bins=200, alpha=0.7)
    axes[-1].scatter([x_init_n], [0], color="black") # show starting point...
    axes[-1].scatter([x_actual_n.mean()], [0], color="blue")
    axes[-1].scatter([x_predicted_n.mean()], [0], color="brown")
  plt.show()
  #plt.savefig("figures/direct1/%d.png" % np.random.randint(0, 10000))
  #plt.close()


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
    compare_predictions_x(init_state, pred_fin_states, fin_states)


def main(fpath, iterations=1):
  assert fpath is not None
  # load the GAN
  gan = load(fpath)
  gan.set_eval(True)
  # define sampling function
  sample_step = get_sample_step(gan)
  def sample_steps(state):
    ans = state
    for i in range(iterations):
      ans = sample_step(ans)
    return ans
  # get comparison data
  init_states, fin_states = get_continuation_dataset(10, 10000, gan.config, iterations=iterations)
  init_states, fin_states = init_states.to(torch.float32), fin_states.to(torch.float32)
  print(fin_states.shape, init_states.shape)
  # compare!
  eval_sample_step(sample_steps, init_states, fin_states)




if __name__ == "__main__":
  from sys import argv
  main(*argv[1:], iterations=ITERATIONS)






