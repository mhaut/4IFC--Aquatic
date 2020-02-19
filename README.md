# 4th International Field Camp Aquatic Ecosystem (BEJA)

This code is part of:

```
J. A. Gallardo, M. E. Paoletti, J. M. Haut, R. Fernandez-Beltran, J. Plaza and A. Plaza.
GPU Parallel Implementation of Dual-Depth Sparse Probabilistic Latent Semantic Analysis for Hyperspectral Unmixing. 
IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing
DOI: 10.1109/JSTARS.2019.2934011
Accepted for publication, 2019.
```

## Example of use
### Download datasets

```
./retrieveData.sh
```

### Install dependencies
```
conda install matplotlib scipy scikit-learn
pip install pycuda
```
#### Install CUDA
If you need install cuda
```
install cuda 10.0 from https://developer.nvidia.com/cuda-10.0-download-archive
add to bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export PATH=/usr/local/cuda/bin:$PATH
source ~/.bashrc
```

### Run code
```
# CUDA VERSION
python -u CUDA_plsa_python.py pLSA samson 0 --seed 0
# NO CUDA VERSION
python NOcuda_plsa_python.py
```
