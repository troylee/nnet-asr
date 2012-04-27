#!/bin/bash

##########################################################
#It's better to run this example line-by-line 
#while looking into the scripts
exit 0
##########################################################



#first of all edit the TIMIT location
vim prepare_timit/prepare_timit.sh



#bypass the locale settings
export LC_ALL=C

#now let's run the sctipt
cd prepare_timit/
bash prepare_timit.sh || exit 1
cd -

#lets join the features of the training set (for faster copy to /tmp)
#(can be used for crossvalidation se too)
./tjoiner.sh || exit 1
#build CRBE-HAMM-DCT transform and perform mean-variance normalization
./tnorm.sh || exit 1
#now train the network
./tnet_train.CPU.sh || exit 1; mv weights weights.CPU
./tnet_train.GPU.sh || exit 1; mv weights weights.GPU
#finally decode the test set by HVite
./decode.sh || exit 1

