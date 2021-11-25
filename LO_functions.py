from thewalrus import perm
import numpy as np
from scipy.stats import unitary_group


def from_Fock_to_conf(fock_state):
    temp_conf = []
    for ix, nphot in enumerate(fock_state):
        temp_conf += [ix for _ in range(nphot)]
    return np.array(temp_conf)


def output_Fock_amplitude(in_fock, out_fock, U):
    out_conf = from_Fock_to_conf(out_fock)
    in_conf = from_Fock_to_conf(in_fock)
    sub_U = U[np.ix_(out_conf, in_conf)]
    perm_val = perm(sub_U)
    norm = np.sqrt(np.prod([np.math.factorial(el) for el in in_fock]) *
                   np.prod([np.math.factorial(el) for el in out_fock]))
    return perm_val/norm


if __name__ == '__main__':
    U = unitary_group.rvs(3) # ranodm unitary in n=3 modes

    in_fock = np.array([1, 0, 1])
    out_fock = np.array([0, 1, 1])
    print(U)

    io_ampl = output_Fock_amplitude(in_fock, out_fock, U)
    print('\n\n calculated amplitude:', io_ampl)
    print('calculated probability:', np.abs(io_ampl)**2)


