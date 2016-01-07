#!/bin/bash
#python main_learn_tag_hdf.py -tf mfcc -l 3 -fm 64



# python main_learn_tag_hdf.py -ne 5 -op adam
#python main_learn_tag_hdf.py -ne 8 -op sgd
python main_learn_tag_hdf.py -ne 3 -op adagrad -act lrelu
python main_learn_tag_hdf.py -ne 3 -op adagrad -act prelu
python main_learn_tag_hdf.py -ne 3 -op adagrad -act elu

python main_learn_tag_hdf.py -ne 3 -op sgd -act lrelu
python main_learn_tag_hdf.py -ne 3 -op sgd -act prelu
python main_learn_tag_hdf.py -ne 3 -op sgd -act elu
#python main_learn_tag_hdf.py -ne 5 -op adadelta
#python main_learn_tag_hdf.py -ne 5 -op rmsprop

