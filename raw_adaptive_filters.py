import numpy as np


def raw_ls(d, u, ir_len=12, f_delay=None, sigma=0., rcond=0.):
    """
    implements ls method of optimization
    and filters u sequence

    :param d: reference signal
    :param u: input signal
    :param ir_len: impulse response length
    :param f_delay: desired delay of filter
    :param sigma: regularization Rxx + (sigma**2)*np.eye(f_ord)
    :param rcond: singular values constraint like pinv
    :return: w - resulting filter coefficients, y - filter output
    """

    if f_delay is None:
        f_delay = ir_len // 2 - 1

    u = np.pad(u, (ir_len - f_delay - 1, f_delay), mode='constant', constant_values=(0, 0))
    U = np.array([u[i:i + ir_len] for i in range(len(d))])
    Uh = U.conj().T

    # find filter weights by ls with regularization
    w = np.linalg.pinv((Uh @ U) + (sigma**2) * np.eye(ir_len), rcond=rcond) @ Uh @ d

    # filter output
    y = U @ w

    return w, y


def raw_lms(d, u, ir_len=12, f_delay=None, lr=None, w_init=None):
    """
    implements lms method of optimization
    and filters u sequence

    :param d: reference signal
    :param u: input signal
    :param ir_len: impulse response length
    :param f_delay: desired delay of filter
    :param lr: learning rate
    :param w_init: initialization of filter weights
    :return: w - resulting filter coefficients, y - filter output
    """
    if f_delay is None:
        f_delay = ir_len // 2 - 1

    if w_init is None:
        w_init = np.random.rand(ir_len) / ir_len + 1j * np.random.rand(ir_len) / ir_len
    w = w_init.astype(complex)

    u = np.pad(u, (ir_len - f_delay - 1, f_delay), mode='constant', constant_values=(0, 0))
    y = np.zeros(len(d), dtype=complex)

    for i in range(len(d)):
        y[i] = u[i:i + ir_len] @ w
        err = d[i] - y[i]
        w = w + lr * err * u[i:i + ir_len].conj()

    return w, y


def raw_rls(d, u, ir_len=12, f_delay=None, sigma_init=10e3, w_init=None):
    """
    implements rls method of optimization
    and filters u sequence

    :param d: reference signal
    :param u: input signal
    :param ir_len: impulse response length
    :param f_delay: desired delay of filter
    :param sigma_init: initialization P = sigma_init*np.eye(ir_len, dtype=complex)
    :param w_init: initialization of filter weights
    :return: w - resulting filter coefficients, y - filter output
    """
    if f_delay is None:
        f_delay = ir_len // 2 - 1

    if w_init is None:
        w_init = np.random.rand(ir_len) / ir_len + 1j * np.random.rand(ir_len) / ir_len
    w = w_init.astype(complex)

    u = np.pad(u, (ir_len - f_delay - 1, f_delay), mode='constant', constant_values=(0, 0))

    y = np.zeros(len(d), dtype=complex)
    P = sigma_init * np.eye(ir_len, dtype=complex)

    for i in range(len(d)):
        ui = u[i:i + ir_len]
        y[i] = ui @ w
        err = d[i] - y[i]
        K = (P @ ui.conj())/(1 + ui@P@ui.conj())
        P -= K.reshape((ir_len, 1))@ui.reshape((1, ir_len))@P
        w += K*err

    return w, y
