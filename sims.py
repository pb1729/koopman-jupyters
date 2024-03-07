from threading import Thread
from queue import Queue

import torch



def vvel_lng_batch(x, v, a, drag, T, dt, nsteps, device="cuda"):
    """ Langevin dynamics with Velocity-Verlet.
    Batched version: compute multiple trajectories in parallel
    This function mutates the position (x) and velocity(v) arrays.
    Compute nsteps updates of the system: We're passed x(t=0) and v(t=0) and
    return x(dt*nsteps), v(dt*nsteps). Leapfrog updates are used for the intermediate steps.
    a(x) is the acceleration as a function of the position.
    drag[] is a vector of drag coefficients to be applied to the system.
    T is the temperature in units of energy.
    dt is the timestep size.
    Shapes:
    x: (batch, coorddim)
    v: (batch, coorddim)
    drag: (coorddim,)
    a: (-1, coorddim) -> (-1, coorddim)
    return = x:(batch, coorddim), v:(batch, coorddim)"""
    assert nsteps >= 1
    assert x.shape == v.shape and drag.shape == x.shape[1:]
    assert x.dtype == torch.float64 and v.dtype == torch.float64
    sqrt_hlf = 0.5**0.5
    noise_coeffs = torch.sqrt(2*drag*T*dt) # noise coefficients for a dt timestep
    def randn():
        return torch.randn(*x.shape, device=device, dtype=torch.float64)
    v += 0.5*(dt*(a(x) - drag*v) + sqrt_hlf*noise_coeffs*randn())
    for i in range(nsteps - 1):
        x += dt*v
        v += dt*(a(x) - drag*v) + noise_coeffs*randn()
    x += dt*v
    v += 0.5*(dt*(a(x) - drag*v) + sqrt_hlf*noise_coeffs*randn())
    return x, v


class TrajectorySim:
    def __init__(self, acc_fn, drag, T, delta_t, t_res):
        """ Object representing a physical system for which we can generate trajectories.
        acc_fn : function defining the system, gives acceleration given position
        drag : vector of drag coefficients, also gives the shape of the position vector
        T : temperature
        delta_t : time spacing at which we take samples
        t_res : time resolution, number of individual simulation steps per delta_t """
        self.acc_fn = acc_fn
        self.drag = drag.to("cuda")
        self.T = T
        self.delta_t = delta_t
        self.t_res = t_res
        self.dt = delta_t/t_res
        self.dim = drag.flatten().shape[0]
    def generate_trajectory(self, batch, N, x=None, v=None):
        """ generate trajectory from initial conditions x, v
            WARNING: initial condition tensors *will* be overwritten
            x: (batch, self.dim)
            v: (batch, self.dim)
            x_traj: (batch, N, self.dim)
            v_traj: (batch, N, self.dim) """
        x_traj = torch.zeros((batch, N, self.dim), device="cuda", dtype=torch.float64)
        v_traj = torch.zeros((batch, N, self.dim), device="cuda", dtype=torch.float64)
        if x is None: x = torch.zeros((batch,) + self.drag.shape, dtype=torch.float64, device="cuda")
        if v is None: v = torch.zeros((batch,) + self.drag.shape, dtype=torch.float64, device="cuda")
        for i in range(N):
            vvel_lng_batch(x, v, self.acc_fn, self.drag, self.T, self.dt, self.t_res)
            x_traj[:, i] = x
            v_traj[:, i] = v
        return x_traj, v_traj


# POLYMER (LINEAR CHAIN)
polymer_length = 12

def get_polymer_a(k, n, dim=3):
    """ Get an acceleration function defining a polymer system with n atoms and spring constant k
    Shapes:
    x: (batch, n*dim)
    a: (batch, n*dim) """
    def a(x):
        x = x.reshape(-1, n, dim)
        ans = torch.zeros_like(x)
        ans[:, 1:] += k*(x[:, :-1] - x[:, 1:])
        ans[:, :-1] += k*(x[:, 1:] - x[:, :-1])
        return ans.reshape(-1, n*dim)
    return a


sims = {
    "1D Ornstein Uhlenbeck": TrajectorySim(
        (lambda x: -x),
        torch.tensor([10.], dtype=torch.float64), 1.0,
        3.0, 60
    ),
    "1D Polymer, Ornstein Uhlenbeck": TrajectorySim(
        get_polymer_a(1.0, polymer_length, dim=1),
        torch.tensor([10.]*polymer_length, dtype=torch.float64), 1.0,
        1.0, 20
    ),
}


# DATASET GENERATION
def get_dataset(sim, N, L, t_eql=0, subtract_cm=0, x_only=False, x_init=None, v_init=None):
  """ generate data from a simulation sim.
      generates N trajectories of length L, ans has shape (N, L, 2*system_dim)
      t_eql is the number of delta_t to wait to system to equilibriate, data before this is thrown away
      subtract_cm: 0 means don't subtract center of mass, positive number n means subtract center of mass
      and assume spatial dimension is n
      x_only: only return the x coordinate and not velocity
      initial_state: x_init, v_init: (N, state_dim)
      if these are not None, then te  """
  x_traj, v_traj = sim.generate_trajectory(N, L+t_eql, x=x_init, v=v_init)
  x_traj, v_traj = x_traj[:, t_eql:], v_traj[:, t_eql:]
  if subtract_cm > 0:
    x_tmp = x_traj.reshape(N, L, -1, subtract_cm)
    x_tmp = x_tmp - x_tmp.mean(2, keepdim=True)
    x_traj = x_tmp.reshape(N, L, -1)
  if x_only: return x_traj
  return torch.cat([
      x_traj,
      v_traj
    ], dim=2)


def dataset_gen(sim, N, L, t_eql=0, subtract_cm=0, x_only=False):
  """ generate many datasets in a separate thread
      t_eql, subtract_cm, x_only all do the same thing as they do in get_dataset() """
  data_queue = Queue(maxsize=32) # we set a maxsize to control the number of items taking up memory on GPU
  def thread_main():
    while True: # queue maxsize stops us from going crazy here
      data_queue.put(get_dataset(sim, N, L, t_eql=t_eql, subtract_cm=subtract_cm, x_only=x_only).to(torch.float32))
  t = Thread(target=thread_main)
  t.start()
  while True:
    data = data_queue.get()
    yield data



# COMPUTE THE THEORETICAL EIGENVALUE #1 FOR THE 1D PROCESS
# v' = -x - drag*v + noise
# if we average out the noise, and set v' = 0, we get: v = -x/drag
# x' = v = -x/drag
# x = exp(-t/drag)
def get_ou_eigen_1d():
  sim = sims["1D Ornstein Uhlenbeck"]
  return torch.exp(-sim.delta_t/sim.drag[0]).item()

# similar, but now v is -kx/drag, where k is associated with a Rouse mode
# we'll let the user of this function worry about the value of k, though
def get_poly_eigen_1d():
  sim = sims["1D Polymer, Ornstein Uhlenbeck"]
  return torch.exp(-sim.delta_t/sim.drag[0]).item()

if __name__ == "__main__":
  print("Linear eigenstate decay per time delta_t for 1D Ornstein Uhlenbeck:", get_ou_eigen_1d())
  print("Linear eigenstate decay per time delta_t for Polymer (assuming reference of k=1):", get_poly_eigen_1d())





