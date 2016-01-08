#!/bin/bash

#python main_learn_tag_hdf.py -ne 10 -op adagrad -act elu
python main_learn_tag_hdf.py -ne 30 -op rmsprop -act elu
python main_learn_tag_hdf.py -ne 30 -op adadelta -act elu
python main_learn_tag_hdf.py -ne 30 -op adam -act elu

python main_learn_tag_hdf.py -ne 30 -op adagrad -act prelu
python main_learn_tag_hdf.py -ne 30 -op rmsprop -act prelu
python main_learn_tag_hdf.py -ne 30 -op adadelta -act prelu
python main_learn_tag_hdf.py -ne 30 -op adam -act prelu

 
python main_learn_tag_hdf.py -ne 30 -op adagrad -act lrelu
python main_learn_tag_hdf.py -ne 30 -op rmsprop -act lrelu
python main_learn_tag_hdf.py -ne 30 -op adadelta -act lrelu
python main_learn_tag_hdf.py -ne 30 -op adam -act lrelu

