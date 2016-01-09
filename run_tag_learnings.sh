#!/bin/bash
python main_learn_tag_hdf.py -ne 5 -op adam -act elu
python main_learn_tag_hdf.py -ne 5 -l 6 -op adam -act elu

# check if otheres are bottlneck
python main_learn_tag_hdf.py -ne 5 -l 5 -op adam -fm 80
python main_learn_tag_hdf.py -ne 5 -l 5 -op adam -lfc 1 -nu 1024
python main_learn_tag_hdf.py -ne 5 -l 4 -op adam -lf rmse

python main_learn_tag_hdf.py -ne 5 -op rmsprop -act elu
python main_learn_tag_hdf.py -ne 30 -op sgd -act elu
