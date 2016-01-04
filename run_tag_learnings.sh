#!/bin/bash
#python main_learn_tag_hdf.py -tf mfcc -l 3 -fm 64


#python main_learn_tag_hdf.py -tf cqt -l 5
#python main_learn_tag_hdf.py -tf cqt -l 6
python main_learn_tag_hdf.py -tf stft -l 5 -ne 45
python main_learn_tag_hdf.py -tf stft -l 6 -ne 45

