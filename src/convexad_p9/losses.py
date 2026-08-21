# =============================================================================
# LOSSES
# =============================================================================
import tensorflow as tf

def mae(Iobs, Icalc, eps=1e-12):
    """
    Mean Absolute Error (normalized).
    """    
    return tf.reduce_sum(tf.abs(Iobs - Icalc), axis = (1,2,3))/tf.reduce_sum(Iobs)

def poisson_kl(Iobs, Icalc, eps=1e-12):
    """
    Poisson KL divergence (per-pixel average).
    """
    ratio = Iobs / (Icalc + eps)
    kl = Icalc - Iobs + tf.math.xlogy(Iobs, ratio)

    N = tf.cast(tf.reduce_prod(tf.shape(Iobs)[1:]), tf.float32)
    return tf.reduce_sum(kl, axis = (1,2,3)) / N
    
def fourier_loss(support, amplitude, phase, Iobs, metric = 'mae'):
    """
    Forward model + MAE data fidelity.
    """
    modulus = support * amplitude

    if isinstance(phase, tuple):
        c, s = phase
        obj = tf.complex(modulus * c, modulus * s)
    else:
        obj = tf.complex(modulus * tf.cos(phase),
                         modulus * tf.sin(phase))

    # --- Padding ---
    obj_shp = tf.shape(obj)
    obs_shp = tf.shape(Iobs)

    dD = obs_shp[0] - obj_shp[1]
    dH = obs_shp[1] - obj_shp[2]
    dW = obs_shp[2] - obj_shp[3]

    pD0 = dD // 2; pD1 = dD - pD0
    pH0 = dH // 2; pH1 = dH - pH0
    pW0 = dW // 2; pW1 = dW - pW0

    obj_p = tf.pad(obj, [[0,0],[pD0,pD1],[pH0,pH1],[pW0,pW1]])

    # --- FFT ---
    Icalc = tf.abs(tf.signal.ifftshift(
        tf.signal.fft3d(tf.signal.fftshift(obj_p, axes=(1,2,3))),
        axes=(1,2,3),
    ))**2

    # --- Prepare data ---
    Iobs = tf.cast(Iobs[tf.newaxis], tf.float32)
    Icalc = tf.cast(Icalc, tf.float32)

    if metric == 'mae': 
        return mae(Iobs, Icalc)
    
    if metric == 'poisson':
        return poisson_kl(Iobs, Icalc)
    else:
        raise ValueError(f"Unknown metric: {metric}, please choose either 'mae' or 'poisson'. ")


def tv_loss_phase(phase, eps=1e-9):
    """
    Total variation for:
    - scalar wrapped phase
    - phasor (c, s)

    Returns per-batch loss.
    """

    def diff(x, axis):
        return tf.experimental.numpy.diff(x, axis=axis)

    # ---------------- PHASOR ----------------
    if isinstance(phase, tuple):
        c, s = phase

        dcx = diff(c, 1); dcy = diff(c, 2); dcz = diff(c, 3)
        dsx = diff(s, 1); dsy = diff(s, 2); dsz = diff(s, 3)

        dcx2 = tf.square(dcx[:, :, :-1, :-1])
        dcy2 = tf.square(dcy[:, :-1, :, :-1])
        dcz2 = tf.square(dcz[:, :-1, :-1, :])

        dsx2 = tf.square(dsx[:, :, :-1, :-1])
        dsy2 = tf.square(dsy[:, :-1, :, :-1])
        dsz2 = tf.square(dsz[:, :-1, :-1, :])

        grad_sq = (dcx2 + dcy2 + dcz2) + (dsx2 + dsy2 + dsz2)

        return tf.reduce_mean(grad_sq + eps, axis = (1,2,3))

    # ---------------- WRAPPED PHASE ----------------
    else:
        phi = phase

        def wrapped_diff(p, axis):
            d = tf.experimental.numpy.diff(p, axis=axis)
            return tf.atan2(tf.sin(d), tf.cos(d))

        dx = wrapped_diff(phi, 1)
        dy = wrapped_diff(phi, 2)
        dz = wrapped_diff(phi, 3)

        dx2 = tf.square(dx[:, :, :-1, :-1])
        dy2 = tf.square(dy[:, :-1, :, :-1])
        dz2 = tf.square(dz[:, :-1, :-1, :])

        grad_sq = dx2 + dy2 + dz2

        return tf.reduce_mean(grad_sq + eps, axis = (1,2,3))


def small_support(support):
    'Penalty on the size of the support'
    return tf.reduce_sum(support, axis = (1,2,3))


@tf.function
def total_loss(
    support,
    amplitude,
    phase,
    Iobs,
    alpha=0.8,
    beta=0.1,
    metric = 'mae',
):
    """
    Full loss combining Fourier fidelity, support size and phase TV

    Parameters
    ----------
    alpha     : weight for support-size penalty
    beta      : weight for phase TV penalty

    """
    fourier = fourier_loss(
        support, amplitude, phase, Iobs, metric = metric)
    small        = alpha * small_support(support)
    smooth_phase = beta  * tv_loss_phase(phase)
    
    return fourier + small + smooth_phase 