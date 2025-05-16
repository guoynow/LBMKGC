python Main_LBMKGC.py -dataset MKG-W -margin 16 -epoch 1000 -save MKG-W-checkpoint -learning_rate 1e-5

python Main_LBMKGC.py -dataset MKG-Y -margin 24 -epoch 1250 -save MKG-Y-checkpoint -learning_rate 1e-5

python Main_LBMKGC.py -dataset DB15K -margin 12 -epoch 1250 -save DB15K-checkpoint -learning_rate 2e-5
