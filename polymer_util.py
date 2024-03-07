import numpy as np
from sims import get_poly_eigen_1d


def rouse(n, length):
  """ get nth Rouse mode for polymer of a given length """
  return np.cos(n*np.pi*(0.5 + np.arange(length))/length) - 0.5*(n == 0)


def tica_theory(n, polymer_length):
  return np.exp(np.log(get_poly_eigen_1d())*4*(np.sin(0.5*np.pi*n/polymer_length))**2)


if __name__ == "__main__":
  # test functionality by plotting the rouse modes...
  import matplotlib.pyplot as plt
  poly_len = 12
  x = np.arange(0, poly_len)
  for n in range(poly_len):
    plt.scatter(x, rouse(n, poly_len))
    plt.show()


# ARBITRARY QUANTA THEORY (THE IDEALISTIC THEORY)
def get_log_expected_singular_values(log_vals, n, show_ans_tups=False):
    """ get the expected singular values where arbitrary numbers
        of quanta can be put into a given mode
        log_vals: logarithms of the value for 1 quanta in a given mode.
                  Should be sorted in increasing order!
        n: we'll compute the largest (least negative) n singular values """
    def H(tup): # energy function
        return sum([tup[i]*log_vals[i] for i in range(dim)])
    dim = len(log_vals)
    ans_tups = [tuple([0]*dim)]
    for i in range(n - 1):
        # get a list of candidates for the next largest tuple
        # a candidate must not be an existing tuple
        # all candidates are increments of some existing tuple
        candidate_tups = []
        for tup in ans_tups:
            for j in range(dim):
                lst = list(tup)
                lst[j] += 1
                inc_tup = tuple(lst)
                if inc_tup not in ans_tups:
                    candidate_tups.append(inc_tup)
                    break
        # find and add to the list the largest of the candidates
        i_best = 0
        H_best = -np.inf
        for i, tup in enumerate(candidate_tups):
            if H(tup) > H_best:
                i_best = i
                H_best = H(tup)
        ans_tups.append(candidate_tups[i_best])
    if show_ans_tups:
        for tup in ans_tups: print(tup)
    return [H(tup) for tup in ans_tups]

def get_n_quanta_theory(num_values, polymer_length):
    log_lin_S = np.log(tica_theory(np.arange(1, polymer_length), polymer_length))
    return np.exp(get_log_expected_singular_values(log_lin_S, num_values))



