#!/bin/bash

##$ -N TNET_EXAMPLE
#$ -o _$JOB_NAME.out
#$ -e _$JOB_NAME.err
#$ -cwd

########################################################
# USE THIS FOR GPU TRAINING
#$ -q long.q@@pco203,long.q@pcspeech-gpu
#$ -l gpu=1



if [ $SGE_O_WORKDIR ]; then cd $SGE_O_WORKDIR; fi

hostname

########################################################
# SELECT TNET LOCATION
#TNET_ROOT='/mnt/matylda5/iveselyk/Tools/TNet/'
TNET_ROOT='/mnt/matylda5/iveselyk/DEVEL/TNet_PACKAGE/TNet/trunk/'

############################################
# CONFIG
TOPOLOGY=(368 1024 1024 1024)
RBM_UNIT_TYPES=(gauss bern bern bern)
#NEGBIAS=1

SCP_TRAIN='/mnt/matylda5/iveselyk/DATABASE/TIMIT/lists/fbank23_TJoiner/fbank23_train_tjoiner15.scp'
FEATURE_TRANSFORM='tr_23Tcontext31_Ham_dct16.transf'
FRM_EXT=15
#CONF=''

#LEARNRATE=0.1
#LEARNRATE_LOW=0.001
#MOMENTUM=0.5
#MOMENTUM_HIGH=0.9
#WEIGHTCOST=0.0002
ITERS=2
ITERS_HIGH=3
SAVEPOINTS=5

#BUNCHSIZE=128
#CACHESIZE=32768
#RANDOMIZE=TRUE

############################################




#create weight directory
[ -d weightsRBM ] || mkdir weightsRBM

#copy the feature transform
cp $FEATURE_TRANSFORM weightsRBM/tr
RBM_transf=weightsRBM/tr


#pretrain layers
for N in $(seq 0 $((${#TOPOLOGY[@]}-2))); do
  #initialize RBM
  python $TNET_ROOT/tools/init/gen_rbm_init.py \
   --dim=${TOPOLOGY[$N]}:${TOPOLOGY[$((N+1))]} \
   --gauss ${NEGBIAS:+ --negbias} \
   --vistype=${RBM_UNIT_TYPES[$N]} \
   --hidtype=${RBM_UNIT_TYPES[$((N+1))]} \
   > weightsRBM/L$N.init
  #make a copy for training 
  RBM=weightsRBM/L$N
  cp weightsRBM/L$N.init $RBM

  #train single RBM layer
  (
   source $TNET_ROOT/tools/train/rbm_training_scheduler.sh
  )
  
  #add RBM to feature transform
  {
   cat $RBM_transf 
   python $TNET_ROOT/tools/rbm2mlplayer/rbm2mlplayer.py $RBM -
  } >${RBM_transf}_L$N
  RBM_transf=${RBM_transf}_L$N
done





