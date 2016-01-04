#!/bin/bash
#python main_learn_tag_hdf.py -tf mfcc -l 3 -fm 64


python main_learn_tag_hdf.py -tf cqt -l 4 -ne 25
python main_learn_tag_hdf.py -tf cqt -l 5 -ne 30
python main_learn_tag_hdf.py -tf cqt -l 6 -ne 35

python main_learn_tag_hdf.py -tf stft -l 4 -ne 25
python main_learn_tag_hdf.py -tf stft -l 5 -ne 30
python main_learn_tag_hdf.py -tf stft -l 6 -ne 45

