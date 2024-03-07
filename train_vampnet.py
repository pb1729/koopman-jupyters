import torch

from sims import sims, get_dataset, dataset_gen, polymer_length
from vampnets import make_VAMPNet, KoopmanModel, vamp_score, res_layer_weight_decay


MODELDIR = "models/"


def train_VAMPNet(model, data_generator, steps, lr=0.0003, weight_decay=None):
    """ train a VAMPNet for a particular system
    dataset - array of trajectories, shape is (N, L, dim)
    N - number of trajectories to create
    L - length of each trajectory """
    optimizer = torch.optim.Adam(model.parameters(), lr)
    stepcount = 0
    for data in data_generator:
        N, L, dim = data.shape
        for t in range(L - 1):
            input_coords_0 = data[:, t]
            input_coords_1 = data[:, t + 1]
            model.zero_grad()
            chi_0 = model.forward(input_coords_0)
            chi_1 = model.forward(input_coords_1)
            loss = -vamp_score(chi_0, chi_1)
            if weight_decay is not None:
                loss = loss + res_layer_weight_decay(model, weight_decay)
            print("   step", stepcount, ": loss =", loss.item())
            loss.backward()
            optimizer.step()
            stepcount += 1
        if stepcount >= steps:
            break
    return model


def main(run_name, *dims):
  dims = [int(dim) for dim in dims]
  print("generating polymer dataset for generating final projection matrices...")
  dataset_poly_fin = get_dataset(sims["1D Polymer, Ornstein Uhlenbeck"], 80000, 80, t_eql=120, subtract_cm=1, x_only=True).to(torch.float32)
  print("done.")
  data_generator = dataset_gen(sims["1D Polymer, Ornstein Uhlenbeck"], 4096, 64, t_eql=120, subtract_cm=1, x_only=True)
  for outdim in dims:
    model = make_VAMPNet(polymer_length, outdim)
    print("\n  OUTDIM = %d" % outdim)
    train_VAMPNet(model, data_generator, 2048, lr=0.0024/outdim)
    train_VAMPNet(model, data_generator, 1024, lr=0.0012/outdim)
    train_VAMPNet(model, data_generator, 1024, lr=0.0006/outdim)
    print("calculating model transforms...")
    kmod = KoopmanModel.fromdata(model, dataset_poly_fin)
    print("saving model...")
    kmod.save("%s%s_%d.koop.pt" % (MODELDIR, run_name, outdim))
    print("done!")


if __name__ == "__main__":
  from sys import argv
  main(*argv[1:])


# TODO: use tensorboard to display training statistics...



