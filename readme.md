# 内容
![alt text](./img/image-4.png)
![alt text](./img/image-3.png)
![alt text](./img/image-2.png)
![alt text](./img/image-1.png)
![alt text](./img/image.png)


[数据集](https://pan.baidu.com/s/1N10URTnXCaWbBWlRLISAWg?pwd=zvjs)
    
* plant-pathology-2021-fgvc8介绍
    * https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8/overview/description

* 论文
    * Arcfaceloss可用于优化细粒度分类问题，论文和MindSpore 1的参考代码
    * https://arxiv.org/abs/1801.07698


# 运行结果
* LeNet5，30 epoch
    * ![alt text](./img/image-5.png)
    * ![alt text](./img/image-6.png)


# 模型
1. LeNet
    * https://blog.csdn.net/qq_43307074/article/details/126022041

    * [模型解析](https://blog.csdn.net/muye_IT/article/details/123539199)

2. VGG16
    * https://blog.csdn.net/m0_50127633/article/details/117045008
    

# 使用GPU
https://blog.csdn.net/m0_51302496/article/details/138013760
```shell

nvidia-smi # cuda 版本

pip uninstall torch torchvision torchaudio 


conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0  pytorch-cuda=11.8 -c pytorch -c nvidia # CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
