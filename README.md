# Tensorflow GPU installation on Ubuntu 22
Install GPU enabled Tensorflow on Ubuntu 22
## Ubuntu drivers and build packages
```bash
sudo ubuntu-drivers install
sudo apt install build-essential libffi-dev pkg-config cmake
sudo apt install zlib1g-dev libssl-dev libsqlite3-dev
```
## NVIDIA related packages
Make sure the driver installed is the Ubuntu tested and verified one.
![Screenshot from 2023-02-27 11-46-22](https://user-images.githubusercontent.com/37543656/221555859-99025c67-c3da-457e-bc91-d27ff899f313.png)
It should support at least CUDA 11:
```bash
$ nvidia-smi    
+-----------------------------------------------------------------------------+
  NVIDIA-SMI 525.78.01    Driver Version: 525.78.01    CUDA Version: 12.0
```
Download CUDA 11.1 and CUDA 11.2 and install them. They should be both in `/usr/local`. The former is needed for `libnvrtc.so.11.1` which seems to hardcoded somewhere in TensorFlow. The latter is needed for `libcudnn8` `11.2`. They should both be in the path:
```bash
export PATH=$PATH:/usr/local/cuda-11.2/bin:/usr/local/cuda-11.1/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.2/lib64:/usr/local/cuda-11.2/extras/CUPTI/lib64:/usr/local/cuda-11.1/lib64:/usr/local/cuda-11.1/extras/CUPTI/lib64
```
Download [TensorRT7](https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/7.2.3/local_repos/nv-tensorrt-repo-ubuntu1804-cuda11.1-trt7.2.3.4-ga-20210226_1-1_amd64.deb). It will contain `libcudnn8` and `libcudnn8-dev`, amongst others. These require CUDA 11.2. Unpack the package with `dpkg`. Install `libcudnn8`:
```bash
sudo dpkg -i libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb
sudo dpkg -i libcudnn8-dev_8.1.1.33-1+cuda11.2_amd64.deb
```
It also contains `libnvinfer7` and `libnvinfer_plugin7`, which should be installed. Create a folder for `libnvinfer7` and its dependencies:
```bash
$mkdir /usr/local/tensorrt7
```
and copy the libraries into it:
```bash
$sudo cp /lib/x86_64-linux-gnu/libnvinfer.so.7.2.3 /usr/local/tensorrt7/
$sudo cp /lib/x86_64-linux-gnu/libnvinfer_plugin.so.7.2.3 /usr/local/tensorrt7/
$sudo cp /lib/x86_64-linux-gnu/libmyelin.so.1.1.116 /usr/local/tensorrt7/
```
Create simlinks to these libraries:
```bash
$sudo ln -s libnvinfer.so.7.2.3 libnvinfer.so.7
$sudo ln -s libnvinfer_plugin.so.7.2.3 libnvinfer_plugin.so.7
$sudo ln -s libmyelin.so.1.1.116 libmyelin.so.1
```
Add the directory to the path (probably unnecessary):
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.2/lib64:/usr/local/cuda-11.2/extras/CUPTI/lib64:/usr/local/cuda-11.1/lib64:/usr/local/cuda-11.1/extras/CUPTI/lib64:/usr/local/tensorrt7
```
## Python
Download and install Python 3.10. The installation should be local:
```bash
./configure --prefix=/home/user/ --enable-optimizations
make
make install
```
Create a Python environment:
```bash
$/home/user/bin/python3 -m venv tfenv
```
and activate it:
```bash
$source tfenv/bin/activate
```
Update `pip` and `setuptools`. Install `wheel`:
```bash
(tfenv)$pip install --upgrade pip setuptools wheel
```
Install TensorFlow and other packages
```bash
(tfenv)$pip install tensorflow
```
Address some issues:
```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true  # to avoid GPU memory issues
export TF_CPP_MIN_LOG_LEVEL=3  # Only print errors
```
Verify the installation:
```bash
(tfenv)$python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```
A more detailed insight can be provided by running:
```bash
(tfenv)$LD_DEBUG=libs python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))" > called_libs.txt 2>&1
```
The output file, `called_libs.txt` contains a lot of stuff, but we can search for the evidence that the correct libraries have been called:
```bash
 3308:     calling init: /usr/local/cuda-11.2/lib64/libcudart.so.11.0
 3308:     calling init: /usr/local/cuda-11.2/lib64/libcublasLt.so.11
 3308:     calling init: /usr/local/cuda-11.2/lib64/libcublas.so.11
 3308:     calling init: /usr/local/cuda-11.1/lib64/libnvrtc.so.11.1
 3308:     calling init: /usr/local/tensorrt7/libmyelin.so.1
 3308:     calling init: /lib/x86_64-linux-gnu/libcudnn.so.8
 3308:     calling init: /usr/local/tensorrt7/libnvinfer.so.7
 3308:     calling init: /usr/local/cuda-11.2/lib64/libnvrtc.so
 3308:     calling init: /lib/x86_64-linux-gnu/libcuda.so
 3308:     calling init: /usr/local/tensorrt7/libnvinfer_plugin.so.7
 3308:     calling init: /usr/local/cuda-11.2/lib64/libcufft.so.10
 3308:     calling init: /usr/local/cuda-11.2/lib64/libcurand.so.10
 3308:     calling init: /usr/local/cuda-11.2/lib64/libcusolver.so.11
 3308:     calling init: /usr/local/cuda-11.2/lib64/libcusparse.so.11
```
