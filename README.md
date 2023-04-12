# TensorFlow GPU installation on Ubuntu 22
Install GPU enabled Tensorflow on Ubuntu 22. This document assumes that all the NVIDIA libraries are installed manually into the operating system (rather than using `conda` for example) and that the versions are:
```bash
Ubuntu 22.04
TensorFlow 2.10 or 2.11
CUDA 11.8 and 11.1
libcudnn8  8.8
TensorRT 7
Python 3.10
```
## Ubuntu drivers and build packages
```bash
sudo ubuntu-drivers install
sudo apt install build-essential libffi-dev pkg-config cmake
sudo apt install zlib1g-dev libssl-dev libsqlite3-dev
```
## NVIDIA related packages
### Driver
Make sure the driver installed is the Ubuntu tested and verified one.
![Screenshot from 2023-02-27 11-46-22](https://user-images.githubusercontent.com/37543656/221555859-99025c67-c3da-457e-bc91-d27ff899f313.png)
It should support at least CUDA 11:
```bash
$ nvidia-smi    
+-----------------------------------------------------------------------------+
  NVIDIA-SMI 525.78.01    Driver Version: 525.78.01    CUDA Version: 12.0
```
### CUDA
Download CUDA 11.8 and install it:
```bash
sudo ./cuda_11.8.0_520.61.05_linux.run --override
```
The package `.run` should install everything in `/usr/local`, and there should be a config file in `/etc/ld.so.conf.d`. Modify the path in your `.bashrc`:
```bash
export PATH=$PATH:/usr/local/cuda-11.8/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64:/usr/local/cuda-11.8/extras/CUPTI/lib64
```
### CUDNN
Donwload version 8.8 and install it:
```bash
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.8.0.121_1.0-1_amd64.deb
```
This should create a folder in `/var`:
```bash
/var/cudnn-local-repo-ubuntu2204-8.8.0.121/
├── B66125A0.pub
├── cudnn-local-B66125A0-keyring.gpg
├── InRelease
├── libcudnn8_8.8.0.121-1+cuda11.8_amd64.deb
├── libcudnn8-dev_8.8.0.121-1+cuda11.8_amd64.deb
├── libcudnn8-samples_8.8.0.121-1+cuda11.8_amd64.deb
├── Local.md5
├── Local.md5.gpg
├── Packages
├── Packages.gz
├── Release
└── Release.gpg
```
Add the pub key or `apt` will complain every time you run it:
```bash
sudo cp /var/cudnn-local-repo-ubuntu2204-8.8.0.121/cudnn-local-B66125A0-keyring.gpg /usr/share/keyrings/
```
At this point, instead of running `sudo apt install libcudnn8`, install the three `.deb` packages manually:
```bash
sudo dpkg -i /var/cudnn-local-repo-ubuntu2204-8.8.0.121/libcudnn8_8.8.0.121-1+cuda11.8_amd64.deb
sudo dpkg -i /var/cudnn-local-repo-ubuntu2204-8.8.0.121/libcudnn8-dev_8.8.0.121-1+cuda11.8_amd64.deb
sudo dpkg -i /var/cudnn-local-repo-ubuntu2204-8.8.0.121/libcudnn8-samples_8.8.0.121-1+cuda11.8_amd64.deb
```
Check that the libraries have been installed:
```bash
$ sudo dpkg -l | grep -i cudnn
ii  cudnn-local-repo-ubuntu2204-8.8.0.121      1.0-1                                   amd64        cudnn-local repository configuration files
ii  libcudnn8                                  8.8.0.121-1+cuda11.8                    amd64        cuDNN runtime libraries
ii  libcudnn8-dev                              8.8.0.121-1+cuda11.8                    amd64        cuDNN development libraries and headers
ii  libcudnn8-samples                          8.8.0.121-1+cuda11.8                    amd64        cuDNN samples
```
or
```bash
$ apt list --installed | grep -i cudnn
cudnn-local-repo-ubuntu2204-8.8.0.121/now 1.0-1 amd64 [installed,local]
libcudnn8-dev/unknown,now 8.8.0.121-1+cuda11.8 amd64 [installed]
libcudnn8-samples/unknown,now 8.8.0.121-1+cuda11.8 amd64 [installed]
libcudnn8/unknown,now 8.8.0.121-1+cuda11.8 amd64 [installed]
```
### TensorRT
The version 7 requires CUDA 11.1, so install it first. It will live in peace with 11.8 in `/usr/local`. Download [TensorRT7](https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/7.2.3/local_repos/nv-tensorrt-repo-ubuntu1804-cuda11.1-trt7.2.3.4-ga-20210226_1-1_amd64.deb). Unpack the package into `/var` with `dpkg`:
```bash
sudo dpkg -i nv-tensorrt-repo-ubuntu1804-cuda11.1-trt7.2.3.4-ga-20210226_1-1_amd64.deb
```
It contains `libnvinfer7` and `libnvinfer_plugin7`, which should be installed:
```bash
sudo dpkg -i libnvinfer7_7.2.3-1+cuda11.1_amd64.deb
sudo dpkg -i libnvinfer-plugin7_7.2.3-1+cuda11.1_amd64.deb
```
This might produce some `CUDA-11.1` related error messages, but the libraries have been extracted. Create a separate folder for `libnvinfer7` and its dependencies:
```bash
mkdir /usr/local/tensorrt7
```
remove the simlinks:
```bash
sudo rm /lib/x86_64-linux-gnu/libnvinfer.so.7
sudo rm /lib/x86_64-linux-gnu/libnvinfer_plugin.so.7
sudo rm /lib/x86_64-linux-gnu/libmyelin.so.1
```
and move the relevant libraries into the new destination:
```bash
sudo mv /lib/x86_64-linux-gnu/libnvinfer.so.7.2.3 /usr/local/tensorrt7/
sudo mv /lib/x86_64-linux-gnu/libnvinfer_plugin.so.7.2.3 /usr/local/tensorrt7/
sudo mv /lib/x86_64-linux-gnu/libmyelin.so.1.1.116 /usr/local/tensorrt7/
```
Create simlinks to these libraries (in `/usr/local/tensorrt7`):
```bash
sudo ln -s libnvinfer.so.7.2.3 libnvinfer.so.7
sudo ln -s libnvinfer_plugin.so.7.2.3 libnvinfer_plugin.so.7
sudo ln -s libmyelin.so.1.1.116 libmyelin.so.1
```
Update the path:
```bash
export PATH=$PATH:/usr/local/cuda-11.1/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.1/lib64:/usr/local/cuda-11.1/extras/CUPTI/lib64:/usr/local/tensorrt7
```
Now run
```bash
sudo apt --fix-broken install
```
to remove the libraries from `/lib/x86_64-linux-gnu`.
## Python
Download and install Python 3.10. The installation should be local:
```bash
./configure --prefix=/home/<user>/ --enable-optimizations
make
make install
```
Create a Python environment:
```bash
$ /home/<user>/bin/python3 -m venv tfenv
```
and activate it:
```bash
$ source tfenv/bin/activate
```
Update `pip` and `setuptools`. Install `wheel`:
```bash
(tfenv)$ pip install --upgrade pip
(tfenv)$ pip install --upgrade setuptools
(tfenv)$ pip install wheel
```
Install TensorFlow and other packages
```bash
(tfenv)$ pip install tensorflow
```
Address some issues:
```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true  # to avoid GPU memory issues
export TF_CPP_MIN_LOG_LEVEL=3  # Only print errors. When debugging, this should set to 0
```
Verify the installation:
```bash
(tfenv)$ python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```
A more detailed insight can be provided by running:
```bash
(tfenv)$ LD_DEBUG=libs python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))" > called_libs.txt 2>&1
```
The output file, `called_libs.txt` contains a lot of stuff, but we can search for the evidence that the correct libraries have been called:
```bash
calling init: /usr/local/cuda-11.8/lib64/libcublasLt.so.11
calling init: /lib/x86_64-linux-gnu/libcuda.so.1
calling init: /usr/local/cuda-11.8/lib64/libnvrtc.so
calling init: /usr/local/cuda-11.8/lib64/libcublas.so.11
calling init: /usr/local/cuda-11.1/lib64/libnvrtc.so.11.1
calling init: /usr/local/tensorrt7/libmyelin.so.1
calling init: /lib/x86_64-linux-gnu/libcudnn.so.8
calling init: /usr/local/tensorrt7/libnvinfer.so.7
calling init: /usr/local/tensorrt7/libnvinfer_plugin.so.7
```
## Jax
`Jax` requires CUDA version of at least 11.4. Install `Jax` into the Python environment:
```bash
(tfenv)$ pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
Make sure GPU is used:
```bash
(tfenv)$ python -c "from jax.lib import xla_bridge; print(xla_bridge.get_backend().platform)"
gpu
```
Check that the correct CUDA libraries are called:
```bash
LD_DEBUG=libs python -c "import jax; print(jax.numpy.sum(jax.random.normal(jax.random.PRNGKey(0),(1000, 1000))))" > jax_libs.txt 2>&1
```
```bash
calling init: /usr/local/cuda-11.8/lib64/libcudart.so.11.0
calling init: /usr/local/cuda-11.8/lib64/libcublasLt.so.11
calling init: /usr/local/cuda-11.8/lib64/libnvrtc.so
calling init: /usr/local/cuda-11.8/lib64/libcublas.so.11
```
## GPU not detected after resuming from suspension
It looks like sometimes NVIDIA _Unified Virtual Memory_ must be reloaded after waking up, if the GPU is not recognised
```bash
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm
```
