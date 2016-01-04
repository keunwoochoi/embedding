#!/bin/bash
# python main_learn_tag_hdf.py -tf mfcc -l 6


#python main_learn_tag_hdf.py -tf cqt -l 5
#python main_learn_tag_hdf.py -tf cqt -l 6
python main_learn_tag_hdf.py -tf stft -l 5
python main_learn_tag_hdf.py -tf stft -l 6

