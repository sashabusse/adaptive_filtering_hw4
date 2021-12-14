import numpy as np
from scipy import io

from raw_adaptive_filters import raw_ls


def load_data(fname='data/test2_25p0.mat'):
    """
    reads file and makes all the corrections from hw2

    :param Fs: sampling frequency
    :param fname: file name to load data from
    :return: x - system input sequence, y - system output sequence (with all the corrections)
    """
    Fs = 245.76E6
    # << load data >> -------------s------------------------------------------
    mat = io.loadmat(fname)
    x = np.reshape(mat['pdin'], (-1,))
    y = np.reshape(mat['pdout'], (-1,))
    tx_freq = float(mat['tx_freq'])
    rx_freq = float(mat['rx_freq'])
    # ------------------------------------------------------------------------

    # << compensate frequency shift and mean value >> --------------------------
    y = (y - np.mean(y)) * np.exp(-1j * 2 * np.pi * ((tx_freq - rx_freq) / Fs) * np.arange(len(y)))
    # --------------------------------------------------------------------------

    # << cut out piece of output that corresponds to the input >> -------------
    xy_corr = np.abs(np.correlate(y, x, mode='valid')) / len(x)
    st_ind = np.argmax(xy_corr)
    y = y[st_ind: st_ind + len(x)]
    # -------------------------------------------------------------------------

    # << gain control calculation >> ---------------------------------------
    g = y.conj().dot(x) / (y.conj().dot(y))
    y = y * g
    # -------------------------------------------------------------------------

    return x, y


def load_for_non_linear_adaptation(filename):
    """
    :param filename: file to load from
    :return: residual error after linear fir application
    """
    pd_in, pd_out = load_data(fname='data/test4_25p0.mat')
    pd_err = pd_in - pd_out

    d = pd_err
    u = pd_in

    Fs = 245.76E6
    reg_sigma = np.sqrt((10 ** (-2)) * Fs)
    r_cond = 0.01
    w_ls, out_ls = raw_ls(d, u, ir_len=5, f_delay=None, sigma=reg_sigma, rcond=r_cond)

    return pd_in, d-out_ls
