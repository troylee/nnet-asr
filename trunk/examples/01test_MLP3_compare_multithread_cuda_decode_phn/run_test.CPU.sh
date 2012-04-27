#!/bin/bash

#/***************************************************************************
# *   copyright            : (C) 2011 by Karel Vesely,UPGM,FIT,VUT,Brno     *
# *   email                : iveselyk@fit.vutbr.cz                          *
# ***************************************************************************
# *                                                                         *
# *   This program is free software; you can redistribute it and/or modify  *
# *   it under the terms of the APACHE License as published by the          *
# *   Apache Software Foundation; either version 2.0 of the License,        *
# *   or (at your option) any later version.                                *
# *                                                                         *
# ***************************************************************************/

if [ ! -r TNet ]; then
  if [ ! -r ../../src/TNet ]; then
    echo "TNet binary not compiled yet" 
    exit 1
  fi
  ln -s ../../src/TNet .  
fi

# TEST SETUP

SCP='lib/test.scp'
INIT='weights/test_mlp.init_weights'
LABELS='lib/test_3s.mlf'
PHONEME_STATES='lib/mono_state_phn_set_135_phn'
TRANSFORM='lib/Hamm_dct_norm'

if [ ! -d $(dirname $INIT) ]; then
  mkdir -p $(dirname $INIT)
fi
if [ ! -e $INIT ]; then
  python ../../tools/init/gen_mlp_init.py --dim=598:1024:135 --gauss --negbias > $INIT
fi

BUNCHSIZE=960 #size of block per-which update is performed
              #lower number leads to faster convergence 
              #of GD but leads to longer epochs

# order of parallelization
THREADS=1 #the BUNCHSIZE is internally divided by number of threads
          #each thread computes own gradient
          #the gradients are summed and weights are updated


{
  hostname
  arch
  date
  echo

  ./TNet -A -D -V -T 021 \
   -H $INIT \
   -I $LABELS -L '*/' -X lab\
   -S $SCP \
   -m $PHONEME_STATES \
   -n 0.008 \
   --TARGETMMF='weights/test_mlp.epoch1-MULTI-THREAD' \
   --BUNCHSIZE=$BUNCHSIZE \
   --CACHESIZE=14400 \
   --RANDOMIZE=TRUE \
   --SEED=123 \
   --THREADS=$THREADS \
   --FEATURETRANSFORM=$TRANSFORM \
   --STARTFRMEXT=25 \
   --ENDFRMEXT=25 \
   2>&1 

  echo
  date
} | tee $0.log

