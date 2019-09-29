# Attention-based Sampler in TASN (Trilinear Attention Sampling Network)

It is an implemetation of attention-based sampler in TASN. 

It's based on [MobulaOP](https://github.com/wkcn/mobulaop), and you don't need to re-build MXNet.

In addition, the implementation of attention-based sampler is available for MXNet and PyTorch.

## Usage
1. Install MobulaOP
```bash
# Clone the project
git clone https://github.com/wkcn/MobulaOP

# Enter the directory
cd MobulaOP

# Install Third-Party Library
pip install -r requirements.txt

# Build
sh build.sh

# Add MobulaOP into Enviroment Variable `PYTHONPATH`
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

2. Clone TASN project
```bash
git clone https://github.com/researchmm/tasn
cd tasn/tasn-mxnet/example/tasn
```

3. Clone this project
```bash
git clone https://github.com/wkcn/AttentionSampler
```
The directory shows as follow:
```bash
├─AttentionSampler
│   ├── attention_sampler
│   ├── imgs
│   └── test.py
├── common
├── data
├── init.sh
├── install.sh
├── model
├── model.py
├── readme
├── train.py
└── train.sh
```

4. Copy the following code on the head of `model.py` of TASN 
```python
import mxnet as mx
import mobula
from attention_sampler import attsampler_mx
mobula.op.load('./AttentionSampler/attention_sampler')
```

You can train TASN model now. Enjoy it!

If this project is helpful, Hope to follow [me](https://github.com/wkcn) and star [the MobulaOP project](https://github.com/wkcn/mobulaop).

Thank you!


Reference Paper
---------------
```
@inproceedings{zheng2019looking,
  title={Looking for the Devil in the Details: Learning Trilinear Attention Sampling Network for Fine-grained Image Recognition},
  author={Zheng, Heliang and Fu, Jianlong and Zha, Zheng-Jun and Luo, Jiebo},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5012--5021},
  year={2019}
}
```
