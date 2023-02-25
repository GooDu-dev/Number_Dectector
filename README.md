# Number_Dectector
This repository is made for junior machine learning developer to learn a basic CNN project.

Incase you want to try my model I have built `Model.hdf5` for testing.

Here is what I used
* tensorflow-gpu 1.13.2
* keras 2.2.4
* opencv-python 3.2.0.8
* cuDNN and CUDA [ This you need to compare with your GPU ]

***

## Train a number detector with tensorflow-gpu 1.13.3 on window 10
1Install cuDNN, CUDA, tensorflow-gpu, keras, opencv-python, matplotlib__
    1.1 I recommend python version 3.6.12 and work with [Anaconda](https://www.anaconda.com/products/distribution)__
    1.2 Install CUDA v10.0__
    1.3 install cuDNN v8.6.0__

**install python 3.6.12 in anaconda**
```
conda create -n your_env pip python=3.6.12
```
```
activate your_env
```

**install tensorflow-gpu, keras, opencv-python, matplotlib 1.13.2**__
*Do not download tensorflow*
```
pip install --upgrade tensorflow-gpu==1.13.2
```
```
pip install --upgrade keras==2.2.4
```
```
pip install --upgrade opencv-python==3.2.0.8
```
```
pip install --upgrade matplotlib
```

**Test your installation**
```
python
```
```
import tensorflow as tf
```
```
hello = tf.constants('hello, tensroflow')
```
```
tf.Session()
```
if tensorflow can detect your GPU. There is a text similar to `Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3003 MB memory) -> physical GPU (device: 0, name: NVIDIA GeForce GTX 1050, pci bus id: 0000:01:00.0, compute capability: 6.1)`

2. Create a packages / folders
create a directory like this
```
--Dectect_Number
    |
    |-- test
    |   |-- 1
    |   |-- 2
    |-- train
    |   |-- 1
    |   |-- 2
    |-- main.py
```

3. run model
```python main.py```