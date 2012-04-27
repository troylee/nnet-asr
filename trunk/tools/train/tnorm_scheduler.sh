#!/bin/bash

#threads were not specified, assume using CUDA
if [[ ! $THREADS ]]; then
  CUDA=1
fi
TNorm=$TNET_ROOT/src/TNorm${CUDA:+Cu}


if [ -r "$FEATURE_TRANSFORM" ]; then
  echo "Using already normalized feature transform: $FEATURE_TRANSFORM"
else
  #Generate the transform, when not supplied by MMF
  if [ ! $MMF ]; then
    MMF="tr_${DIM_FEA}Tcontext$((2*FRM_EXT + 1))_Ham_dct${DCT_BASE}"
    if [ ! -e $MMF ]; then
      python $TNET_ROOT/tools/transform/gen_hamm_dct.py \
      --dimIn=$DIM_FEA \
      --startFrmExt=$FRM_EXT \
      --endFrmExt=$FRM_EXT \
      --dctBaseCnt=$DCT_BASE \
      > $MMF
    fi
  fi

  #output file with cmn cvn normalization
  NORM=${MMF}${STK_CONF:+_$(basename ${STK_CONF} .conf)}.norm

  #compute the normalization
  $TNorm -D -A -T 1 -S $SCP_TNORM ${STK_CONF:+ -C $STK_CONF} -H $MMF \
    --TARGET-MMF=$NORM \
    --START-FRM-EXT=$FRM_EXT \
    --END-FRM-EXT=$FRM_EXT 

  if [ $? != 0 ]; then echo error occured; exit 1; fi

  #assemble the transform from MMF and normalization
  FEATURE_TRANSFORM=$(basename $NORM .norm).transf
  cat $MMF $NORM > $FEATURE_TRANSFORM

  echo "TNorm finished OK"
fi


