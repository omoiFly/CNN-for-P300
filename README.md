#  CNN for P300 Evoked Potentials

### Description

[Dataset Description](http://www.bbci.de/competition/iii/desc_II.pdf)

|          | Subject_A | Subject_B |
| -------- | --------- | --------- |
| Accuracy | 0.87      | 0.95      |

Using PyTorch for CNN Model

### Usage

download [processed data](https://pan.baidu.com/s/1Tmh2D4oyL8PXKg8y-pI0ow) 提取码: hw8b 

to dataset/ folder

run train.py ( Simply modify 'A' to 'B' for different subject )

### Model

See Model.py  'class Vanilla'

### Dataset

Average every 15 times repeat experiment for higher SNR.

Because every character only creates 2 positive samples but 10 negative, just repeat the 2 positive sample 4 more times ( See Dataset.py ) in order to balance positive between negative.

dataset folder contains only processed data.

### Optimizer

SGD + momentum with lr=5e-4 momentum=0.9 weight_decay=1e-4
