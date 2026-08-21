# =============================================================================
# PHASE PARAMETERIZATION
# =============================================================================
import tensorflow as tf

class PhaseParam(tf.keras.layers.Layer):
    def initialize(self, batch_size, grid_shape, initial_guess=None):
        raise NotImplementedError


class GridPhase(PhaseParam):
    """Direct phase parameterization."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._batch_size = None
        self._grid_shape = None
        self._initial_guess = None

    def initialize(self, batch_size, grid_shape, initial_guess=None):
        self._batch_size = batch_size
        self._grid_shape = grid_shape
        self._initial_guess = initial_guess
        self.build()

    def build(self, input_shape=None):
        if self._grid_shape is None:
            return
        if self._initial_guess is not None:
            phase = tf.math.angle(self._initial_guess)
            if len(phase.shape) == 3:
                phase = phase[None, ...]
            initializer = tf.keras.initializers.Constant(phase.numpy())
            shape = phase.shape
        else:
            initializer = tf.keras.initializers.RandomUniform(minval=0.0, maxval=1.0)
            shape = [self._batch_size, *self._grid_shape]
        self.phase = self.add_weight(
            name="phase", shape=shape, dtype=tf.float32,
            trainable=True, initializer=initializer,
        )
        super().build(input_shape)

    def compute_phase(self):
        return self.phase


class GridPhasor(PhaseParam):
    """Unit-circle phasor parameterization."""

    def __init__(self, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self._batch_size = None
        self._grid_shape = None

    def initialize(self, batch_size, grid_shape, initial_guess=None):
        self._batch_size = batch_size
        self._grid_shape = grid_shape
        self.build()

    def build(self, input_shape=None):
        if self._grid_shape is None:
            return
        shape = [self._batch_size, *self._grid_shape]
        self.c_raw = self.add_weight(
            name="c_raw", shape=shape, dtype=tf.float32,
            trainable=True, initializer=tf.keras.initializers.Ones(),
        )
        self.s_raw = self.add_weight(
            name="s_raw", shape=shape, dtype=tf.float32,
            trainable=True, initializer=tf.keras.initializers.Zeros(),
        )
        super().build(input_shape)

    def compute_phasor(self):
        denom = tf.sqrt(self.c_raw**2 + self.s_raw**2 + self.eps)
        return self.c_raw / denom, self.s_raw / denom

class DisplacementPhasor(PhaseParam):
    """Phasor parameterization via longitudinal displacement field:
       phi(r) = |Q_hkl| * u_parallel(r)
       (c, s) = (cos(phi), sin(phi))
    """

    def __init__(self, hkl = [1,1,1], lattice_matrix=None, init_scale=1e-3, **kwargs):
        """
        hkl: iterable of 3 ints [h,k,l]
        lattice_matrix: optional (3x3) real-space lattice matrix (in meters or Å)
                        If None, assumes orthonormal basis (Q = 2π * hkl)
        """
        super().__init__(**kwargs)
        self.hkl = tf.constant(hkl, dtype=tf.float32)
        self.lattice_matrix = lattice_matrix
        self.init_scale = init_scale

        self._batch_size = None
        self._grid_shape = None
        self.Qnorm = None

    def initialize(self, batch_size, grid_shape, initial_guess=None):
        self._batch_size = batch_size
        self._grid_shape = grid_shape

        # Compute |Q_hkl|
        if self.lattice_matrix is not None:
            # Reciprocal lattice: B = 2π * (A^{-1})^T
            A = tf.convert_to_tensor(self.lattice_matrix, dtype=tf.float32)
            B = 2.0 * tf.constant(tf.constant(3.141592653589793, dtype=tf.float32)) * tf.transpose(tf.linalg.inv(A))
            Q = tf.linalg.matvec(B, self.hkl)
        else:
            # Simplified orthonormal case
            Q = 2.0 * tf.constant(3.141592653589793, dtype=tf.float32) * self.hkl

        self.Qnorm = tf.norm(Q)

        self.build(initial_guess)

    def build(self, initial_guess=None):
        shape = [self._batch_size, *self._grid_shape]

        if initial_guess is not None:
            # Convert phase guess → displacement
            phase0 = tf.math.angle(initial_guess)
            if len(phase0.shape) == 3:
                phase0 = tf.tile(phase0[None, ...], [self._batch_size, 1, 1, 1])
            u0 = phase0 / (self.Qnorm + 1e-8)
        else:
            u0 = self.init_scale * tf.random.normal(shape, dtype=tf.float32)

        self.u = self.add_weight(
            name="u_parallel",
            shape=shape,
            dtype=tf.float32,
            trainable=True,
            initializer=tf.keras.initializers.Constant(u0.numpy() if isinstance(u0, tf.Tensor) else u0),
        )

        super().build(None)

    def compute_phase(self):
        return self.Qnorm * self.u

    def compute_phasor(self):
        phi = self.compute_phase()
        return tf.cos(phi), tf.sin(phi)