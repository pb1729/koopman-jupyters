import torch

from sims import sims, dataset_gen
from wgan import GANTrainer, GET_COND, COND_DIM, STATE_DIM, SUBTRACT_MEAN, X_ONLY, SIM
from run_visualization import TensorBoard
from vampnets import KoopmanModel


# other constants:
BATCH = 256



def batchify(dataset_gen, batchsz):
  for dataset in dataset_gen:
    N, L, state_dim = dataset.shape
    assert N % batchsz == 0
    for i in range(0, N, batchsz):
      yield dataset[i:i+batchsz]


def train(gan, save_path):
  """ train a GAN. inputs:
    gan       - a GANTrainer to be fed training data
    save_path - string, location where the model should be saved to
    board     - None, or a TensorBoard to record training progress """
  run_name = ".".join(save_path.split("/")[-1].split(".")[:-1])
  print(run_name)
  board = TensorBoard(run_name)
  data_generator = dataset_gen(SIM, 4096, 64, t_eql=120, subtract_cm=SUBTRACT_MEAN, x_only=X_ONLY)
  for i, trajs in enumerate(batchify(data_generator, BATCH)):
    if i % 4096 == 0:
      print("\nsaving...")
      gan.save(save_path)
      print("saved.\n")
      if i > 100000: break # end training after this many steps plus change
    N, L, state_dim = trajs.shape
    cond = GET_COND(trajs[:, :-1].reshape(N*(L - 1), state_dim))
    data = trajs[:, 1:].reshape(N*(L - 1), state_dim)
    # TODO: this seems good for application of WGAN, but won't work for Koopman operator.
    # Should probably just predict absolute position for now and use subtract_cm
    #data -= cond # we only try and predict the relative change in positon and velocity
    loss_d, loss_g = gan.train_step(data, cond)
    print(f"{i}\t ℒᴰ = {loss_d:05.6f}\t ℒᴳ = {loss_g:05.6f}")
    board.scalar("loss_d", i, loss_d)
    board.scalar("loss_g", i, loss_g)



def main(save_path, load_path=None):
  if load_path is None:
    gan = GANTrainer.makenew(STATE_DIM, COND_DIM)
  else:
    gan = GANTrainer.load(load_path)
  train(gan, save_path)


if __name__ == "__main__":
  from sys import argv
  main(*argv[1:])


