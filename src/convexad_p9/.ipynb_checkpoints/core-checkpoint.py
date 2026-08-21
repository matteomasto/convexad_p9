# =============================================================================
# MODEL
# =============================================================================
import tensorflow as tf
from .support import HalfSpaceSupport
from .phase import GridPhase, GridPhasor, DisplacementPhasor
from .losses import total_loss

class PhaseRetrievalModel(tf.keras.Model):
    """Joint support + phase model."""

    def __init__(self, I, batch_size, phase_type='grid', grid_shape=None, support_kwargs=None, phasor_kwargs = None):
        super().__init__()
        self.I = I
        self.batch_size = batch_size

        if grid_shape is None:
            D, H, W = I.shape
            grid_shape = (D // 2, H // 2, W // 2)
        self.grid_shape = grid_shape

        # Instantiate sub-layers
        self.supporter = HalfSpaceSupport(**(support_kwargs or {}))
        if phase_type == "grid":
            self.phaser = GridPhase()
        elif phase_type == "phasor":
            self.phaser = GridPhasor()
        elif phase_type == 'displacement':
            self.phaser = DisplacementPhasor(**(phasor_kwargs or {}))
            
        else:
            raise ValueError(f"Unknown phase_type: {phase_type}")

    def build(self, input_shape=None):
        self.supporter.initialize(self.batch_size, self.grid_shape)
        self.phaser.initialize(self.batch_size, self.grid_shape)
        super().build(input_shape)

    def call(self, inputs=None):
        support = self.supporter.compute_support()
        Iobs = tf.cast(self.I[None, ...], tf.float32)
        N = tf.reduce_prod(tf.shape(Iobs)[1:])
        sum_I = tf.reduce_sum(Iobs, axis=(1, 2, 3), keepdims=True)
        sum_S = tf.reduce_sum(support**2, axis=(1, 2, 3), keepdims=True)
        amplitude = tf.sqrt(sum_I / (tf.cast(N, tf.float32) * sum_S + 1e-12))

        if hasattr(self.phaser, "compute_phasor"):
            return support, amplitude, self.phaser.compute_phasor()
        else:
            return support, amplitude, self.phaser.compute_phase()


# =============================================================================
# TRAINING
# =============================================================================

@tf.function
def train_step(
    model,
    Iobs,
    optimizer,
    alpha_small=0.0,
    beta_tv=0.0,
    noise_scale=0.0,
    metric = 'mae',
):
    """
    Training step with:
    - manifold constraint (S² normals)
    - optional Langevin noise
    """

    with tf.GradientTape(persistent=True) as tape:
        support, amplitude, phase = model()

        loss = total_loss(
            support,
            amplitude,
            phase,
            Iobs,
            alpha=alpha_small,
            beta=beta_tv,
            metric = metric,
        )

    vars_main = model.trainable_variables
    grads = tape.gradient(loss, vars_main)

    # ---- Langevin noise ----
    if noise_scale > 0.0:
        grads = [
            g + tf.random.normal(tf.shape(v), stddev=noise_scale)
            if g is not None else None
            for g, v in zip(grads, vars_main)
        ]

    # ---- Riemannian projection for support normals ----
    hs = model.supporter

    new_grads = []
    for g, v in zip(grads, vars_main):
        if g is None:
            new_grads.append(None)
        elif v is hs.n:
            new_grads.append(hs.project_tangent(g))
        else:
            new_grads.append(g)

    optimizer.apply_gradients(zip(new_grads, vars_main))

    # ---- Retraction to S² ----
    hs.retract()

    return loss
