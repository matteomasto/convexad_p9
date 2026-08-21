# convexad-p9

<img width="1240" height="820" alt="image" src="https://github.com/user-attachments/assets/df4366b5-9a40-466b-a82e-a13eff3cfae0" />

---

Geometrically regularized automatic differentiation framework for BCDI phase retrieval on Power9 / PPC64LE systems.

This repository provides the PPC64LE / Power9 implementation of ConvexAD, designed for HPC systems using the OpenCE TensorFlow stack.

The API and package structure are intentionally kept aligned with the main [`convex-ad`](https://github.com/matteomasto/convex-ad) implementation.

---

## Requirements

- **Linux PPC64LE / Power9**
- **Python 3.10**
- **TensorFlow 2.10.1**
- **CUDA 11.4**
- **cuDNN 8.3**
- **NCCL 2.12**

The reference environment uses the following stack:

| Package | Version | Source |
|---|---:|---|
| `python` | 3.10.14 | conda-forge |
| `tensorflow` | 2.10.1 | OpenCE |
| `tensorflow-base` | 2.10.1 | OpenCE |
| `tensorflow-estimator` | 2.10 | OpenCE |
| `keras` | 2.10.0 | OpenCE |
| `cudatoolkit` | 11.4.4 | OpenCE |
| `cudnn` | 8.3.0 | OpenCE |
| `nccl` | 2.12.7 | OpenCE |
| `numpy` | 1.26.4 | conda-forge |
| `scipy` | 1.13.1 | conda-forge |
| `scikit-image` | 0.25.0 | conda-forge |
| `matplotlib` | 3.10.0 | conda-forge |
| `hdf5plugin` | 3.3.1 | conda-forge |

> **Note:** TensorFlow and the CUDA runtime are not installed by `pip`. They are expected to be provided by a compatible Power9/OpenCE environment.

---

## Installation

### Step 1 — Activate a compatible environment

Activate a Power9 environment containing TensorFlow 2.10.1 and CUDA 11.4:

```bash
conda activate <your-p9-environment>
```

Verify the TensorFlow version:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

Expected output:

```
2.10.1
```

### Step 2 — Install from GitHub

```bash
pip install git+https://github.com/matteomasto/convexad-p9.git
```

This installs `convexad-p9` and its Python dependencies while leaving the existing TensorFlow/OpenCE installation untouched.

#### Alternative: install from source

If you want to develop or modify the code, or access the example notebooks directly, clone the repository:

```bash
git clone https://github.com/matteomasto/convexad-p9.git
cd convexad-p9
pip install -e .
```

This installs the package in editable mode and keeps the `examples/` and `notebooks/` folders available in your working directory.

### Verify the installation

```python
import convexad_p9

print(convexad_p9.__version__)  # should print 0.1.0
```

### Verify GPU access

Before running a reconstruction, verify that TensorFlow can access the Power9 GPU:

```python
import tensorflow as tf

print(tf.__version__)
print(tf.config.list_physical_devices("GPU"))
```

Expected output is similar to:

```
2.10.1
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

If the list is empty, check the cluster CUDA environment and the TensorFlow/OpenCE installation before proceeding.

---

## Running the example notebooks

### Step 1 — Get the example files

If you installed via `pip install git+...`, the notebook files are not copied to your machine. Retrieve them by cloning the repository separately:

```bash
git clone https://github.com/matteomasto/convexad-p9.git
cd convexad-p9
```

### Step 2 — Install Jupyter

```bash
pip install jupyter
```

or, for JupyterLab:

```bash
pip install jupyterlab
```

### Step 3 — Register the environment as a Jupyter kernel

This ensures the notebook uses the same Python environment where `convexad_p9` and TensorFlow are installed:

```bash
python -m ipykernel install --user \
    --name=convexad-p9 \
    --display-name "Python (convexad-p9)"
```

### Step 4 — Launch Jupyter

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

Navigate to the `examples/` or `notebooks/` folder and open the desired `.ipynb` file, then go to:

**Kernel → Change kernel → Python (convexad-p9)**

---

## Usage

See the `examples/` and `notebooks/` folders for complete reconstruction workflows.

```python
import numpy as np
import convexad_p9

from convexad_p9.core import PhaseRetrievalModel, train_step
from convexad_p9.losses import total_loss

# Load your diffraction data
Iobs = np.load("data.npz")["I"].astype(np.float32)

# Build the phase retrieval model
model = PhaseRetrievalModel(
    Iobs,
    batch_size=4,
)

# ... see examples/ and notebooks/ for a complete workflow
```

---

## Package structure

```
convexad_p9/
├── __init__.py
├── core.py      # PhaseRetrievalModel and optimization
├── losses.py    # Data fidelity and regularization losses
├── phase.py     # Phase parameterizations
├── support.py   # Support parameterizations
├── utils.py     # Numerical utilities
└── viz.py       # Visualization utilities
```

---

## Dependencies

The package intentionally does not install TensorFlow or CUDA dependencies automatically.

The main Python dependencies are:

| Package | Version |
|---|---|
| `numpy` | >= 1.26, < 2 |
| `scipy` | >= 1.13, < 1.14 |
| `scikit-image` | >= 0.25, < 0.26 |
| `matplotlib` | >= 3.10, < 3.11 |
| `hdf5plugin` | >= 3.3, < 4 |

The reference accelerator stack is:

| Component | Version |
|---|---|
| TensorFlow | 2.10.1 |
| CUDA | 11.4 |
| cuDNN | 8.3 |
| NCCL | 2.12 |

These accelerator dependencies should be provided by the HPC/OpenCE environment.

---

## Relationship to the x86 implementation

`convexad-p9` is the Power9/PPC64LE implementation of ConvexAD.

| Platform | Package | Repository |
|---|---|---|
| x86_64 + NVIDIA CUDA | `convex_ad` | [convex-ad](https://github.com/matteomasto/convex-ad) |
| PPC64LE / Power9 + NVIDIA CUDA | `convexad_p9` | [convexad-p9](https://github.com/matteomasto/convexad-p9) |

The two implementations share the same package structure and API where possible, while targeting different TensorFlow/CUDA environments.

---

## License

See [LICENSE](LICENSE).