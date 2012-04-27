#!/bin/bash
#This script header contains directives for the SGE (Sun Grid Engine).

##$ -N TNET_EXAMPLE
#$ -o _$JOB_NAME.out
#$ -e _$JOB_NAME.err
#$ -cwd

########################################################
# USE THIS FOR GPU TRAINING (DIABLED ##$)
##$ -q long.q@@pco203,long.q@pcspeech-gpu
##$ -l gpu=1

########################################################
# USE THIS FOR MULTITHREAD-CPU TRAINING (ENABLED #$)
#$ -q long.q@@stable
#$ -pe smp 7
THREADS=6 #<<<COMMENT THIS TO RUN GPU TRAINING



if [ $SGE_O_WORKDIR ]; then cd $SGE_O_WORKDIR; fi

hostname

########################################################
# SELECT TNET LOCATION
TNET_ROOT='../../'



########################################################
# FEATURES CONFIG 
#
# set LIST and counts to prepare SCP files on the fly
#LIST='../01TJoiner/ihmtrain07.v1_1.crbe.vtln.rand.good_parts_crbe_cmn_cvn_noExt_tjoiner25.scp_head10k'
#train_start=0
#train_count=9000
#cv_start=9000
#cv_count=1000
#SCP_CV='lists/cv.scp'
#SCP_TRAIN='lists/train.scp'

# or set existing SCP_CV SCP_TRAIN lists
SCP_CV='prepare_timit/workdir/lists/cv_fea.scp'
SCP_TRAIN='train_fea_tjoiner15.scp'


########################################################
# THIS IS THE SED ARGUMENT THAT WILL CHANGE DISTANT 
# FILE LOCATION TO LOCAL FILE LOCATION!!!!!
#COPY_LOCAL=1 #DISABLED(we don't know the root directory)
SED_ARG_CREATE_LOCAL='s|mnt/[^\/]*|tmp|g' #UNUSED
###################

# OTHER CONFIG
MLF_CV='prepare_timit/workdir/mlfs/ref.mlf'
MLF_TRAIN='prepare_timit/workdir/mlfs/ref.mlf'
PHONELIST='prepare_timit/workdir/dicts/dict'
FEATURE_TRANSFORM='tr_23Tcontext31_Ham_dct16.transf'
FRM_EXT=15

LEARNRATE=4.0
#MOMENTUM=0.5 #only with GPU
#WEIGHTCOST=1e-6 #only with GPU

# END OF CONFIG
########################################################




########################################################
#copy data to local
#(or just split the lists if COPY_LOCAL!=1)
source $TNET_ROOT/tools/train/copy_local.sh

########################################################
#clean the data upon exit
trap "source $TNET_ROOT/tools/train/delete_local.sh" EXIT


########################################################
#run the TNet training

#inintialize the network
NN_INIT='nnet_368_500_39.init'
if [ ! -e $NN_INIT ]; then
  {
    # more commands here...
    python $TNET_ROOT/tools/init/gen_mlp_init.py --dim=368:500:39 --gauss --negbias
    # more commands here...
  } > $NN_INIT 
else
  echo using preinitialized network $NN_INIT
fi 


#run the training
source $TNET_ROOT/tools/train/training_scheduler.sh

