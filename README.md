# HAET-2021-competition-baseline-code

# Main Guidelines

The aim of the competition is to achieve the best possible accuracy in a classification task given a limited training time. The best three teams will receive awards and will be invited to present their solution during the workshop.

In more details, the training will be performed on an Nvidia V100 GPU running with an Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz processor, with 12GB of memory, with an allowed training time of 10 minutes.

# Instructions to run our code:

Have python 3.6+ installed
pip install -r requirements.txt
python training.py
python test.py --net_sav HAET_model.pt


