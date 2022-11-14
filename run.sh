#!/bin/bash
# lap14(example)

  #Dependency based Location-aware transformation
python ./train_dep.py --dataset lap14 --num_epoch 100 --learning_rate 0.001 --repeats 5
#python ./train_dep.py --dataset lap14 --num_epoch 100 --learning_rate 0.001 --tree True --repeats 5


  #SE-attention based Location-aware transformation
#python ./train_se.py --dataset lap14 --num_epoch 100 --learning_rate 0.0001 --repeats 5
#python ./train_se.py --dataset lap14 --num_epoch 100 --learning_rate 0.0001 --tree True --repeats 5
