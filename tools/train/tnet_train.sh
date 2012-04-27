#!/bin/bash

##$ -N TNET_EXAMPLE
#$ -o _$JOB_NAME.out
#$ -e _$JOB_NAME.err
#$ -cwd

########################################################
# USE THIS FOR GPU TRAINING
##$ -q long.q@@pco203,long.q@pcspeech-gpu
##$ -l gpu=1

########################################################
# USE THIS FOR MULTITHREAD-CPU TRAINING
#$ -q long.q@@stable
#$ -pe smp 7
THREADS=6



if [ $SGE_O_WORKDIR ]; then cd $SGE_O_WORKDIR; fi

hostname
echo "Job: $JOB_ID, Task: $SGE_TASK_ID"

########################################################
# SELECT TNET LOCATION
#TNET_ROOT='/mnt/matylda5/iveselyk/Tools/TNet/'
TNET_ROOT='/mnt/matylda5/iveselyk/DEVEL/TNet_PACKAGE/TNet/trunk/'



########################################################
# FEATURES CONFIG 
#
# set LIST and counts to prepare on the fly
#LIST='../01TJoiner/ihmtrain07.v1_1.crbe.vtln.rand.good_parts_crbe_cmn_cvn_noExt_tjoiner25.scp_head10k'
#N=$(cat $LIST | wc -l)
#train_start=0
#train_count=$((N*9/10))
#cv_start=$((N*9/10))
#cv_count=$((N*1/10))
#SCP_CV='lists/cv.scp'
#SCP_TRAIN='lists/train.scp'

# or set existing SCP_CV SCP_TRAIN lists
SCP_CV='/mnt/matylda5/iveselyk/DATABASE/TIMIT/lists/fbank23_TJoiner/fbank23_cv_tjoiner15.scp'
SCP_TRAIN='/mnt/matylda5/iveselyk/DATABASE/TIMIT/lists/fbank23_TJoiner/fbank23_train_tjoiner15.scp'

# FEATURE NORMALIZATION
#STK_CONF='norm.conf'


########################################################
# THIS IS THE SED ARGUMENT THAT WILL CHANGE DISTANT 
# FILE LOCATION TO LOCAL FILE LOCATION!!!!!
COPY_LOCAL=1
SED_ARG_CREATE_LOCAL='s|mnt/scratch05|tmp|g'
###################

# OTHER CONFIG
MLF_CV='/mnt/matylda5/iveselyk/DATABASE/TIMIT/labels/ref_3s.mlf'
MLF_TRAIN='/mnt/matylda5/iveselyk/DATABASE/TIMIT/labels/ref_3s.mlf'
PHONELIST='/mnt/matylda5/iveselyk/DATABASE/TIMIT/labels/monostates_117'
FEATURE_TRANSFORM='tr_23Tcontext31_Ham_dct16.transf'
FRM_EXT=15

LEARNRATE=4.0
#MOMENTUM=0.5 #only with GPU
#WEIGHTCOST=1e-6

#BUNCHSIZE=512
#CACHESIZE=16384
#RANDOMIZE=TRUE
#TRACE=5 #01..progress 02..dots_on_bunch 04..profile
#TNET_FLAGS=" " # -A..cmdline -D..config -V verbose
#CONFUSIONMODE=max #no,max,soft,dmax,dsoft [in CPU TNet only]

#MAX_ITER=20
#MIN_ITER=1
#KEEP_LRATE_ITER=0
#END_HALVING_INC=0.1
#START_HALVING_INC=0.5
#HALVING_FACTOR=0.5

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
NN_INIT='nnet_368_2105_117.init'
if [ ! -e $NN_INIT ]; then
  {
    # more commands here...
    python $TNET_ROOT/tools/init/gen_mlp_init.py --dim=368:2105:117 --gauss --negbias
    # more commands here...
  } > $NN_INIT 
else
  echo using preinitialized network $NN_INIT
fi 


#run the training
source $TNET_ROOT/tools/train/training_scheduler.sh


