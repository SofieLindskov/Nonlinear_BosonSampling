#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 13:48:07 2021

@author: sofielindskov
"""

from thewalrus import perm
import numpy as np
#from scipy.stats import unitary_group
import math 
import cmath 
from itertools import product
import matplotlib.pyplot as plt

def from_Fock_to_conf(fock_state):
    temp_conf = []
    for ix, nphot in enumerate(fock_state):
        temp_conf += [ix for _ in range(nphot)]
    return np.array(temp_conf)


def output_Fock_amplitude(in_fock, out_fock, U):
    if np.sum(in_fock) == 0 and np.sum(out_fock) == 0:
        return 1.
    elif np.sum(in_fock) != np.sum(out_fock):
        return 0.
    out_conf = from_Fock_to_conf(out_fock)
    in_conf = from_Fock_to_conf(in_fock)
    sub_U = U[np.ix_(out_conf, in_conf)]
    #print('in_fock, out_fock, U', in_fock, out_fock, U)
    #print('(sub_U.real, sub_U.imag, shape)', (sub_U.real, sub_U.imag, sub_U.shape))
    if sub_U.shape == (1,1):
        perm_val = sub_U[0,0]
    else:
        perm_val = perm(sub_U)
    norm = np.sqrt(np.prod([np.math.factorial(el) for el in in_fock]) *
                   np.prod([np.math.factorial(el) for el in out_fock]))
    return perm_val/norm
    

def coherent_state(alpha, max_phot=2, normalized = True):
    norm = np.exp(np.abs(alpha)**2 /2)
    state_vec = norm * np.array([alpha**n/np.sqrt(np.math.factorial(n)) for n in range(max_phot+1)])
    if normalized:
        state_vec = state_vec / np.linalg.norm(state_vec)
    return state_vec

def apply_photonsuppr_nonlin(state, nl_eta, normalized = False):
    state_out = state
    if len(state_out) > 1:
        state_out[1] = (1-nl_eta)*state_out[1]
    if normalized:
        state_out = state_out / np.linalg.norm(state_out)
    return state_out 

def states_tensor_prod(states_list):
    phot_nums_lists = [list(range(len(state))) for state in states_list]
    confs_list = list(product(*phot_nums_lists))
    amplitudes = list(map(np.prod,list(product(*states_list))))
    return confs_list, amplitudes
        
    
def get_output_confs(max_num_photons = 2, num_modes = 2):
    return list(product(range(max_num_photons+1), repeat = num_modes))
    
def get_all_output_ampls(output_confs, in_state_dict, U):
    return [sum([in_state_dict[in_conf] * output_Fock_amplitude(in_conf, out_conf, U) for in_conf in in_state_dict])
            for out_conf in output_confs]

if __name__ == '__main__':
    #U2 = unitary_group.rvs(2) # ranodm unitary in n=2 modes
    #Unitary that corresponds to the interferometer.
    phi = np.pi/2
    #phi_2 = np.pi
    #theta = np.pi
    
    U = 0.5*np.array([[1-cmath.exp(1j*phi),
                       1j + 1j*cmath.exp(1j*phi)],
                      [1j + 1j*cmath.exp(1j*phi),
                       1-cmath.exp(1j*phi)]])
    
    U_bs = np.array([[1., 1.j], [1.j, -1.]])*np.sqrt(0.5) 
    def U_phase(phi): 
        return np.array([[1., 0.], [0., np.exp(1.j*phi)]])
    
    def U_MZI(theta, phi):
        return U_bs @ U_phase(theta) @ U_bs @ U_phase(phi)
    
    U = U_MZI(np.pi/2, np.pi/2)

    
    
    #U2 = cmath.exp(1j*phi/2)*np.array([[cmath.exp(1j*phi)*math.cos(theta), 
    #                                    cmath.exp(1j*phi_2)*math.sin(theta)],
    #                                   -cmath.exp(-1j*phi_2)*math.sin(theta),
    #                                   cmath.exp(-1j*phi)*math.cos(theta)])
   
   
    print('U=',U)

    all_in_states = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [0, 2]])
    all_out_states = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [0, 2]])
   
    dim = len(all_in_states)
    H_matrix = np.zeros((dim, dim),dtype='complex')
    for in_idx, in_fock in enumerate(all_in_states):
        for out_idx, out_fock in enumerate(all_out_states):
            # check order!
            H_matrix[in_idx, out_idx] = output_Fock_amplitude(in_fock, out_fock, U)

    io_ampl = output_Fock_amplitude(in_fock, out_fock, U)
    #print('\n\n calculated amplitude:', io_ampl)
    #print('calculated probability:', np.abs(io_ampl)**2)
    
    alpha=0.15
    beta=0.15
    
    nl_eta = 0.4
    
    # defining input coherent states
    alpha_state = coherent_state(alpha, max_phot=2, normalized = True)
    beta_state = coherent_state(beta, max_phot=2, normalized = True)
    
    print('alpha_state', alpha_state)
    print('beta_state', beta_state)
    print()
    
    # applying single-mode non-linearity to the individual coherent states
    alpha_state = apply_photonsuppr_nonlin(alpha_state, nl_eta)
    beta_state = apply_photonsuppr_nonlin(beta_state, nl_eta)
    
    print('alpha_state after NL', alpha_state)
    print('beta_state after NL', beta_state)
    print()
    
    
    # make tensor product
    confs, ampls = states_tensor_prod([alpha_state, beta_state])
    input_full_state_dict = dict(zip(confs, ampls))
    
    print('confs',confs)
    print('ampls',ampls)
    print()
    
    print('full_state_dict', input_full_state_dict)
    print()
    
    
    # define detectable output configurations
    out_confs = get_output_confs(max_num_photons = 2, num_modes = 2)
    print('out_confs', out_confs)
    print()
    
    # evolve state through linear-optics and calculate the amplitudes of the output configurations
    out_confs_ampls = get_all_output_ampls(out_confs, input_full_state_dict, U)
    out_confs_probs = np.abs(np.array(out_confs_ampls))**2
    
    output_full_state_dict = dict(zip(out_confs, out_confs_ampls))
    print('output_full_state_dict', output_full_state_dict)
    
    
    
    # plot all 
    
    
    plt.bar(range(len(out_confs_probs)), out_confs_probs, 1, align='center')
    plt.yscale('log')
    plt.xticks(range(len(out_confs_probs)), out_confs)
    plt.show()
    
   
    