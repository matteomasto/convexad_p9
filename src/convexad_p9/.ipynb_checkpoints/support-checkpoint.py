# =============================================================================
# SUPPORT PARAMETERIZATION
# =============================================================================
import tensorflow as tf

class SupportParam(tf.keras.layers.Layer):
    """Abstract support parameterization."""
    def initialize(self, batch_size, grid_shape):
        raise NotImplementedError
    def compute_support(self):
        raise NotImplementedError


class HalfSpaceSupport(SupportParam):
    """
    Support defined as intersection of N half-spaces.
    S(x) = ∏_i sigmoid((d_i - n_i · x) / eps)
    """

    def __init__(self, N: int, size_factor: float = 4.0, eps: float = 0.4, **kwargs):
        super().__init__(**kwargs)
        self.N = int(N)
        self.size_factor = float(size_factor)
        self.eps_init = eps
        self._batch_size = None
        self._grid_shape = None

    def initialize(self, batch_size, grid_shape):
        self._batch_size = batch_size
        self._grid_shape = grid_shape
        self.build()

    def build(self, input_shape=None):
        if self._grid_shape is None:
            return

        D, H, W = self._grid_shape
        batch_size = self._batch_size

        # Coordinate grid (not a weight, just a fixed tensor)
        z = tf.linspace(-(D - 1) / 2.0, (D - 1) / 2.0, D)
        y = tf.linspace(-(H - 1) / 2.0, (H - 1) / 2.0, H)
        x = tf.linspace(-(W - 1) / 2.0, (W - 1) / 2.0, W)
        zz, yy, xx = tf.meshgrid(z, y, x, indexing='ij')
        coords = tf.stack([xx, yy, zz], axis=-1)
        self.coords = coords[None, ...]  # (1, D, H, W, 3)

        # Radius scaling
        R = tf.cast(tf.reduce_min(self._grid_shape), tf.float32) / tf.cast(self.size_factor, tf.float32)

        # Initial normals and offsets
        n0 = tf.linalg.l2_normalize(tf.random.normal([batch_size, self.N, 3]), axis=-1)
        d0 = tf.ones([batch_size, self.N], dtype=tf.float32) * R

        self.eps = self.add_weight(
            name="eps", shape=(), trainable=False,
            initializer=tf.keras.initializers.Constant(self.eps_init),
        )
        self.n = self.add_weight(
            name="normals", shape=n0.shape, dtype=tf.float32, trainable=True,
            initializer=tf.keras.initializers.Constant(n0.numpy()),
        )
        self.d = self.add_weight(
            name="offsets", shape=d0.shape, dtype=tf.float32, trainable=True,
            initializer=tf.keras.initializers.Constant(d0.numpy()),
        )
        super().build(input_shape)
        
    def compute_support(self):
        coords = self.coords[..., None, :]                      # (1,D,H,W,1,3)
        normals = self.n[:, None, None, None, :, :]             # (B,1,1,1,N,3)
        dot = tf.reduce_sum(coords * normals, axis=-1)          # (B,D,H,W,N)
        d = self.d[:, None, None, None, :]                      # (B,1,1,1,N)
        
        signed_dist = (d - dot) / self.eps                      # (B,D,H,W,N)
        
        # Soft-min over half-spaces (β controls sharpness of min)
        beta = 1.0
        softmin = -1.0/beta * tf.reduce_logsumexp(
            -beta * signed_dist, axis=-1
        )                                                        # (B,D,H,W)
        
        return tf.sigmoid(softmin)

    def project_tangent(self, g):
        inner = tf.reduce_sum(self.n * g, axis=-1, keepdims=True)
        return g - inner * self.n

    def retract(self):
        self.n.assign(tf.math.l2_normalize(self.n, axis=-1))
