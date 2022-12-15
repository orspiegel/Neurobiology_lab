# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
BASELINE = 34.7
#############################################################################
# Data representation
#############################################################################
def get_pulse_max(vec):
    # returns time (as index) at the pulse moment.
    # later, we will normalize the graphs s.t the pulse will be at 0 time.
    pulse_time = np.where(vec == min(vec))[0][0]
    print(pulse_time)
    return pulse_time

def load_mat_as_dict(file):
    mat = scipy.io.loadmat(file)
    mat = {k: v for k, v in mat.items() if k[0] != '_'}
    return mat

def get_X_Y_vectors_data(mat):
    reaction_data = np.array(np.transpose(mat['data']))[0]
    pulse_vec = np.array(np.transpose(mat['data']))[0]
    return (reaction_data, pulse_vec)
    
def get_normalized_values(mat):
    reaction_data, pulse_vec = get_X_Y_vectors_data(mat)
    offset = get_pulse_max(pulse_vec)
    time = np.arange(0, len(reaction_data))
    return (time - offset, reaction_data)

def align(unaligned1, unaligned2):
    pulse_index1, pulse_index2 = get_pulse_max(unaligned1), get_pulse_max(unaligned2)
    mean_vec = []
    # for i in range(-1700, 1000): # for amp and duration
    for i in range(-1000, 450): # for repititions
        mean_vec.append((unaligned1[pulse_index1+i] + unaligned2[pulse_index2+i])/2)
    return mean_vec

def get_one_mean_vec(file1, file2):
    mean_vec = align(get_X_Y_vectors_data(load_mat_as_dict(file1))[0],
                     get_X_Y_vectors_data(load_mat_as_dict(file2))[0])
    mean_vec = baseline_normalization(mean_vec)
    return mean_vec

def plot_mean_of2_sets(file1, file2, volume, units):
    mean_vec = get_one_mean_vec(file1,file2)
    plt.errorbar(np.arange(0, len(mean_vec), 80)-1500, mean_vec[::80], yerr=0.02, label=f'{volume} {units}')
    plt.legend()

def baseline_normalization(mean_vec):
    if mean_vec[0] < BASELINE:
        mean_vec += BASELINE - mean_vec[0]
    elif mean_vec[0] > BASELINE:
        mean_vec -= mean_vec[0] - BASELINE
    return mean_vec

def plot_amp_train():
    units = 'miliampere'
    plot_mean_of2_sets('set_a1_100ma_800ms.mat', 'set_a2_100ma_800ms.mat', 100, units)
    plot_mean_of2_sets('set_b1_80ma_800ms.mat', 'set_b2_80ma_800ms.mat', 80, units)
    plot_mean_of2_sets('set_c1_60ma_800ms.mat', 'set_c2_60ma_800ms.mat', 60, units)
    plot_mean_of2_sets('set_d1_40ma_800ms.mat', 'set_d2_40ma_800ms.mat', 40, units)
    plot_mean_of2_sets('set_e1_30ma_800ms.mat', 'set_e2_30ma_800ms.mat', 30, units)
    plot_mean_of2_sets('set_f1_20ma_800ms.mat', 'set_f2_20ma_800ms.mat', 20, units)
    plt.xlabel('Time (milisec)')
    plt.ylabel('Reaction - baseline location difference')
    plt.show()

def plot_len_train():
    units = 'miliseconds'
    plot_mean_of2_sets('set_g1_100ma_600ms.mat', 'set_g2_100ma_600ms.mat', 600, units)
    plot_mean_of2_sets('set_h1_100ma_400ms.mat', 'set_h2_100ma_400ms.mat', 400, units)
    plot_mean_of2_sets('set_i1_100ma_200ms.mat', 'set_i2_100ma_200ms.mat', 200, units)
    plot_mean_of2_sets('set_j1_100ma_160ms.mat', 'set_j2_100ma_160ms.mat', 160, units)
    plt.xlabel('Time (milisec)')
    plt.ylabel('Reaction - baseline location difference')
    plt.show()

def plot_rep_train():
    units = 'repititions'
    plot_mean_of2_sets('set_k1_1_def_16.mat', 'set_k2_1_def_16.mat', 16, units)
    plot_mean_of2_sets('set_l1_1_def_12.mat', 'set_l2_1_def_12.mat', 12, units)
    plot_mean_of2_sets('set_m1_1_def_8.mat', 'set_m2_1_def_8.mat', 8, units)
    plot_mean_of2_sets('set_n1_1_def_4.mat', 'set_n2_1_def_4.mat', 4, units)
    plt.xlabel('Time (milisec)')
    plt.ylabel('Reaction - baseline location difference')
    plt.show()
#############################################################################
# Data analysis
#############################################################################
def peak_to_peak(reaction_vec):
    p2p_diff = np.ptp(reaction_vec)
    print(BASELINE- np.amin(reaction_vec))
    return p2p_diff

def p2p_each_amp(*mean_vectors):
    signals = [get_one_mean_vec('set_a1_100ma_800ms.mat', 'set_a2_100ma_800ms.mat'),
    get_one_mean_vec('set_b1_80ma_800ms.mat', 'set_b2_80ma_800ms.mat'),
    get_one_mean_vec('set_c1_60ma_800ms.mat', 'set_c2_60ma_800ms.mat'),
    get_one_mean_vec('set_d1_40ma_800ms.mat', 'set_d2_40ma_800ms.mat'),
    get_one_mean_vec('set_e1_30ma_800ms.mat', 'set_e2_30ma_800ms.mat'),
    get_one_mean_vec('set_f1_20ma_800ms.mat', 'set_f2_20ma_800ms.mat')]
    for amp, val in enumerate(signals):
        print(f'Amplitude {amp} p2p diff: ',peak_to_peak(val))


if __name__ == '__main__':
    # Plot of amplitude set:
    #plot_amp_train()
    #plot_len_train()
    p2p_each_amp()


