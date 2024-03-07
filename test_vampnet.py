import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

from sims import sims, get_dataset, polymer_length
from vampnets import KoopmanModel
from polymer_util import tica_theory, get_n_quanta_theory


def main(nm, *dims):
  dims = [int(dim) for dim in dims]
  # plotting setup
  cmap = matplotlib.colormaps['tab10']
  x = np.linspace(0., polymer_length, 100)
  plt.plot(x, tica_theory(x, polymer_length), color="black", linestyle="--", label="theory (up to 1 quanta)")
  plt.scatter(np.arange(85), get_n_quanta_theory(85, polymer_length), color="black", marker="1", label="theory (any # of quanta)")
  # generate a new dataset for testing
  print("generating polymer dataset...")
  dataset_poly = get_dataset(sims["1D Polymer, Ornstein Uhlenbeck"], 6000, 80, t_eql=120, subtract_cm=1, x_only=True).to(torch.float32)
  print("done.")
  # evaluate each model in turn
  for i, sz in enumerate(dims):
    kmod = KoopmanModel.load("models/%s_%d.koop.pt" % (nm, sz))
    print(sz, kmod.S)
    scores = kmod.eval_score(dataset_poly)
    sdim, = kmod.S.shape
    col = cmap(i/10.)
    #plt.scatter(np.arange(1, 1 + sdim), scores.cpu().numpy(), marker=".", color=col, label="d=%d evaluation score" % sz)
    plt.scatter(np.arange(1, 1 + sdim), kmod.S.cpu().numpy(), marker="o", color=col, label="d=%d learned" % sz)
  for i in range(0, polymer_length):
    height = tica_theory(i, polymer_length)
    plt.plot([0., 85.], [height, height], color="lightgray")
  #plt.ylim(0., 1.)
  plt.xlabel("eigenvalue index")
  plt.ylabel("eigenvalue")
  plt.legend()
  plt.show()


if __name__ == "__main__":
  from sys import argv
  main(*argv[1:])


