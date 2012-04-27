#!/bin/bash

#$ -o _$JOB_NAME.out
#$ -e _$JOB_NAME.err
#$ -cwd

##$ -q long.q@@pco203,long.q@pcspeech-gpu
##$ -l gpu=1

if [ $SGE_O_WORKDIR ];then cd $SGE_O_WORKDIR; fi

TNET_ROOT='../../'
 
#########################################################
# SETUP
#
#set the input files
SCP='train_fea_tjoiner15.scp'
#CONF='norm.conf'
#MMF='' 

#set input transform parameters
FRM_EXT=15  #left/right temporal context in frames, total context is FRM_EXT*2+1
DIM_IN=23   #dimensionality of raw features
DCT_BASE=16 #number of bases in DCT transform (including C0)

#
#
#########################################################


#Generate the transform, when not supplied by MMF
if [ ! $MMF ]; then
  MMF="tr_${DIM_IN}Tcontext$((2*FRM_EXT + 1))_Ham_dct${DCT_BASE}"
  if [ ! -e $MMF ]; then
    python $TNET_ROOT/tools/transform/gen_hamm_dct.py \
    --dimIn=$DIM_IN \
    --startFrmExt=$FRM_EXT \
    --endFrmExt=$FRM_EXT \
    --dctBaseCnt=$DCT_BASE \
    > $MMF
  fi
fi

#output file with cmn cvn normalization
NORM=${MMF}${CONF:+_$(basename ${CONF} .conf)}.norm

#compute the normalization
$TNET_ROOT/src/TNorm -D -A -T 1 -S $SCP ${CONF:+ -C $CONF} -H $MMF \
  --TARGET-MMF=$NORM \
  --START-FRM-EXT=$FRM_EXT \
  --END-FRM-EXT=$FRM_EXT || { echo "$0 error occured"; exit; }

#assemble the transform from MMF and normalization
cat $MMF $NORM > $(basename $NORM .norm).transf

echo "TNorm finished OK"

