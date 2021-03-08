# HAET-2021-competition

The aim of the competition is to achieve the best possible accuracy in a classification task given a limited training time. The best three teams will receive awards and will be invited to present their solution during the workshop.

In more details, the training will be performed on an Nvidia V100 GPU running with an Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz processor, with 12GB of memory, with an allowed training time of 10 minutes.

# Abstract

Training of deep Neural networks is considered a time-consuming and computationally intensive task that demands extra high computational resources taking many days. Therefore, it is our interest to reduce the complexity of the training process. We adopted a super-convergence learning rate policy, and mixed-precision training algorithm to accelerate the training phase. We referred to existing work (David Page 2018 https://github.com/davidcpage/cifar10-fast) and fine-tuned hyperparameters to maximize the validation accuracy under the condition that only training in 5000 sampling samples.

# Instructions to run our code:

Have python 3.6+ installed

pip install -r requirements.txt

python training.py

python test.py --net_sav HAET_model.pt


