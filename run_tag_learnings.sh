#!/bin/bash

#python main_learn_tag_hdf.py -ne 10 -op adagrad -act elu
sleep 4h
python main_learn_tag_hdf.py -ne 7 -op rmsprop -act elu
python main_learn_tag_hdf.py -ne 7 -l 4 -op adadelta -act elu
python main_learn_tag_hdf.py -ne 7 -op adam -act elu

python main_learn_tag_hdf.py -ne 7 -l 4 -op adagrad -act prelu
python main_learn_tag_hdf.py -ne 7 -op rmsprop -act prelu
python main_learn_tag_hdf.py -ne 7 -l 4 -op adadelta -act prelu
python main_learn_tag_hdf.py -ne 7 -op adam -act prelu

 
python main_learn_tag_hdf.py -ne 7 -l 4 -op adagrad -act lrelu
python main_learn_tag_hdf.py -ne 7 -op rmsprop -act lrelu
python main_learn_tag_hdf.py -ne 7 -l 4 -op adadelta -act lrelu
python main_learn_tag_hdf.py -ne 7 -op adam -act lrelu

