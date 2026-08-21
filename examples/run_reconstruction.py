import tensorflow as tf
import numpy as np
import scipy
from scipy.fft import fftshift, ifftshift, fftn
import pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import hdf5plugin
import h5py

import convex_ad
from convex_ad.core import *
from convex_ad.viz import *
from convex_ad.losses import *

from tensorflow.signal import fft3d, ifft3d, ifftshift, fftshift
import colorcet as cc
# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
Iobs = np.load("data_prova.npz")["I"].astype(np.float32)

# ------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------

batch_size = 4

support_kwargs = dict(
    N=64,              # number of half-spaces (complexity of support)
    size_factor=4.0,   # initial radius relative to grid
    eps=0.8,           # softness of boundary (critical parameter)
)

model = convex_ad.PhaseRetrievalModel(
    I=Iobs,
    batch_size=batch_size,
    phase_type="phasor",          
    support_kwargs=support_kwargs,
)

# ------------------------------------------------------------
# Optimizer (fixed: AMSGrad)
# ------------------------------------------------------------

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.05,
    decay_steps=500,       
    decay_rate=0.9,        
    staircase=True
)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr_schedule,
    beta_1=0.9,            
    beta_2=0.98,           
    epsilon=1e-6,          
    amsgrad=True,         
)

# ------------------------------------------------------------
# Training hyperparameters
# ------------------------------------------------------------

n_steps = 5000

alpha_small = 0.0   # support sparsity
beta_tv     = 0.0   # phase smoothness
noise_scale = 0.0    # Langevin noise (set >0 if needed)

# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------

for step in range(n_steps):

    loss = convex_ad.train_step(
        model,
        Iobs,
        optimizer,
        alpha_small=alpha_small,
        beta_tv=beta_tv,
        noise_scale=noise_scale,
        metric = 'mae'
    )

    if step % 200 == 0:
        print(f"step={step:5d} | loss={loss.numpy():.6e}")

# ------------------------------------------------------------
# Extract reconstruction
# ------------------------------------------------------------

support, amplitude, phase = model()

idx = np.argmin(convexad.fourier_loss(support, amplitude, phase, Iobs).numpy())

support = support.numpy()[idx]
amplitude = amplitude.numpy()[idx]

if isinstance(phase, tuple):
    c, s = phase
    phase = np.arctan2(s.numpy()[idx], c.numpy()[idx])
else:
    phase = phase.numpy()[idx]

obj = amplitude * support * np.exp(1j * phase)

# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------

from convex_ad import viz
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
    tf.signal.fft3d(tf.signal.fftshift(obj_p))
    
)**2

viz.plot_3D_projections(Icalc, log_scale=True)
viz.plot_2D_slices_middle_only_phase(obj)