
#!/bin/bash

sleep 120
python main_learn_tag_hdf.py -ne 5 -op adam -act elu -l 3 -fm 2 -nu 4 -do 0.0 -do_fc 0.0 -reg 0.0 -reg_fc 0.0 -tf stft
python main_learn_tag_hdf.py -ne 5 -op adam -act elu -l 3 -fm 2 -nu 64 -do 0.0 -do_fc 0.0 -reg 0.0 -reg_fc 0.0 -tf stft
python main_learn_tag_hdf.py -ne 5 -op adam -act elu -l 3 -fm 2 -nu 1024 -do 0.0 -do_fc 0.0 -reg 0.0 -reg_fc 0.0 -tf stft

python main_learn_tag_hdf.py -ne 5 -op adam -act elu -l 5 -fm 16 -nu 32 -do 0.0 -do_fc 0.0 -reg 0.0 -reg_fc 0.0 -tf stft
python main_learn_tag_hdf.py -ne 5 -op adam -act elu -l 5 -fm 32 -nu 128 -do 0.25 -do_fc 0.0 -reg 0.0 -reg_fc 0.00001 -tf stft

