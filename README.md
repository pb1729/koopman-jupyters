# Koopman Operator and Large Δt Learned Approximations of the Transfer Operator


## Vampnets

VampNets learn a reduced dimension approximation of the Koopman operator. These can make useful building blocks/features for various applications. One such application is described in the next section.


## Learned Approximations of the Transfer Operator

Instead of running a simulation for a certain number of time steps, what if we replace it with a neural network forward pass? There is typically some randomness in the dynamics of the system, so we should have a neural network that samples from some probability distribution. I.e. a diffuser architecture or GAN architecture would be a good choice.

Currently all architectures are some variant of WGANs. To train, run something like:

```
python train_wgan.py models/my_model.wgan.pt
```

Configure the training run by editing these lines:

```
config = Config("1D Polymer, Ornstein Uhlenbeck", "wgan_dn",
      cond=Condition.ROUSE, x_only=True, subtract_mean=1,
      batch=8, simlen=16,
      n_rouse_modes=3)
```


## The 65536 Challenge

This challenge is one of being efficient with training data (and to a lesser extent, training time). The parameters of the challenge are as follows:
* Batch size is 8
* The length of each trajectory in the batch is 16
* The number of training steps is 65536

The goal is to get good results (good approximation of the simulation's transfer operator) within these constraints.



