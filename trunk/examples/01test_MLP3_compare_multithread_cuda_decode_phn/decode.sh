#!/bash/bin

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

if [ ! -r TFeaCatCu ]; then
  if [ ! -r ../../src/TFeaCatCu ]; then
    echo "TFeaCatCu binary not compiled yet" 
    exit 1
  fi
  ln -s ../../src/TFeaCatCu .  
fi


NNET='weights/test_mlp.epoch1-CUDA'
dir=decode
if [ ! -d $dir/posteriors ]; then
  mkdir -p $dir/posteriors
fi


{

#generate dictionary
cat lib/mono_state_phn_set_135_phn | cut -d "_" -f 1 | uniq > $dir/monophones45
cat $dir/monophones45 | sed 's/.*/& &/' > $dir/dict
#generate recognition network
HBuild $dir/monophones45 $dir/phoneloop.net
#generate gmmbypass
../../tools/decode/gen_HTK_gmmbypass.sh lib/mono_state_phn_set_135_phn $dir/HTK_gmmbypass.mmf

#generate posteriors
./TFeaCatCu -D -A -T 1 \
 -S lib/test.scp \
 -H $NNET \
 -l $dir/posteriors \
 -y htk_post \
 --FEATURETRANSFORM=lib/Hamm_dct_norm \
 --GMMBYPASS=true \
 --START-FRM-EXT=25 \
 --END-FRM-EXT=25

ls $dir/posteriors/* > $dir/posteriors.scp

#decode
HVite -T 1 -H $dir/HTK_gmmbypass.mmf -S $dir/posteriors.scp -i $dir/test_hyp.mlf -w $dir/phoneloop.net $dir/dict $dir/monophones45

#test accuarcy (cheating on train data)
#prepare reference mlf
cat lib/test_3s.mlf | awk '{if(NF==3){split($3,a,"_");if(phn!=a[1]){ phn=a[1]; print phn;}}else print $0}' > $dir/test_ref.mlf
HResults -I $dir/test_ref.mlf /dev/null  $dir/test_hyp.mlf

} | tee $0.log
