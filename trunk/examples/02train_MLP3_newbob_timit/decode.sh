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

if [ ! -r TFeaCat ]; then
  if [ ! -r ../../src/TFeaCat ]; then
    echo "TFeaCat binary not compiled yet" 
    exit 1
  fi
  ln -s ../../src/TFeaCat .  
fi


NNET=$(ls weights/nnet_368_500_39_final* | tail -n 1)
dir=_decode
if [ ! -d $dir/posteriors ]; then
  mkdir -p $dir/posteriors
fi


{

#generate dictionary
cat prepare_timit/workdir/dicts/dict | cut -d "_" -f 1 | uniq > $dir/monophones39
cat $dir/monophones39 | sed 's/.*/& &/' > $dir/dict
#generate recognition network
HBuild $dir/monophones39 $dir/phoneloop.net
#generate gmmbypass
../../tools/decode/gen_HTK_gmmbypass_1s.sh $dir/monophones39 $dir/HTK_gmmbypass.mmf

#generate posteriors
./TFeaCat -D -A -T 1 \
 -S prepare_timit/workdir/lists/test_fea.scp \
 -H $NNET \
 -l $dir/posteriors \
 -y htk_post \
 --FEATURETRANSFORM='tr_23Tcontext31_Ham_dct16.transf' \
 --GMMBYPASS=true \
 --START-FRM-EXT=15 \
 --END-FRM-EXT=15

ls $dir/posteriors/* > $dir/posteriors.scp

#decode
HVite -T 1 -H $dir/HTK_gmmbypass.mmf -S $dir/posteriors.scp -i $dir/test_hyp.mlf -p -1 -w $dir/phoneloop.net $dir/dict $dir/monophones39

#test accuarcy
#prepare reference mlf
cat lib/test_3s.mlf | awk '{if(NF==3){split($3,a,"_");if(phn!=a[1]){ phn=a[1]; print phn;}}else print $0}' > $dir/test_ref.mlf
HResults -I ./prepare_timit/workdir/mlfs/ref.mlf /dev/null  $dir/test_hyp.mlf

} | tee _$0.log
