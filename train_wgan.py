import torch

from sims import dataset_gen
from run_visualization import TensorBoard
from vampnets import KoopmanModel
from configs import Config, Condition, load, save, makenew


def batchify(dataset_gen, batchsz):
  for dataset in dataset_gen:
    N, L, state_dim = dataset.shape
    assert N % batchsz == 0
    for i in range(0, N, batchsz):
      yield dataset[i:i+batchsz]


def train(gan, save_path):
  """ train a GAN. inputs:
    gan       - a model to be fed training data
    save_path - string, location where the model should be saved to
    board     - None, or a TensorBoard to record training progress """
  run_name = ".".join(save_path.split("/")[-1].split(".")[:-1])
  print(run_name)
  board = TensorBoard(run_name)
  config = gan.config # configuration for this run...
  data_generator = dataset_gen(config.sim, 128*config.batch, config.simlen,
    t_eql=config.t_eql, subtract_cm=config.subtract_mean, x_only=config.x_only)
  for i, trajs in enumerate(batchify(data_generator, config.batch)):
    if i % 512 == 511:
      print("\nsaving...")
      save(gan, save_path)
      print("saved.\n")
      #if i >= 65535: break # end training here
      if i >= 16384: break # end training here
    N, L, state_dim = trajs.shape
    cond = config.cond(trajs[:, :-1].reshape(N*(L - 1), state_dim))
    data = trajs[:, 1:].reshape(N*(L - 1), state_dim)
    loss_d, loss_g = gan.train_step(data, cond)
    print(f"{i}\t ℒᴰ = {loss_d:05.6f}\t ℒᴳ = {loss_g:05.6f}")
    board.scalar("loss_d", i, loss_d)
    board.scalar("loss_g", i, loss_g)



def main(save_path, load_path=None):
  if load_path is None:
    config = Config("1D Polymer, Ornstein Uhlenbeck, medium", "wgan_dn",
      cond=Condition.ROUSE, x_only=True, subtract_mean=1,
      batch=8, simlen=16, t_eql=4,
      n_rouse_modes=3)
    gan = makenew(config)
  else:
    gan = load(load_path)
  train(gan, save_path)


if __name__ == "__main__":
  from sys import argv
  main(*argv[1:])


