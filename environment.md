```bash
conda create -n fssgan python=3.9
```

### 确保系统中安装了 g++、cmake 和 ninja：
```bash
sudo apt-get update
sudo apt-get install build-essential cmake ninja-build
```


### 更新 setuptools、wheel 和 pip：
```bash
pip install --upgrade setuptools wheel pip
```

### 强制安装 CPU 版本的 torchcsprng->安装纯 CPU 版本的 PyTorch
```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cpu
```

### 手动删除残留的构建目录：
```bash
rm -rf ./build ./dist ./torchcsprng.egg-info

pip install -e .
```

### at the .py begining
```bash
import os
os.environ["GLOO_SOCKET_IFNAME"] = 'enp3s0'
os.environ["GLOO_IPV6"] = '0'
```


#lscpu | grep -E "CPU\(s\)|Thread\(s\) per core

```bash
RENDEZVOUS=tcp://192.168.1.2:1242 WORLD_SIZE=2 RANK=0 python test_fastsecnet.py
# succeed! then:
RENDEZVOUS=tcp://192.168.1.2:1242 WORLD_SIZE=2 RANK=0 python multi_train_DCGAN.py
```
