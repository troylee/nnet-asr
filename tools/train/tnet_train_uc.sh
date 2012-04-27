#!/bin/bash

##$ -N TNET_EXAMPLE
#$ -o _$JOB_NAME.out
#$ -e _$JOB_NAME.err
#$ -cwd

########################################################
# USE THIS FOR GPU TRAINING
#$ -q long.q@@pco203,long.q@pcspeech-gpu
#$ -l gpu=1

########################################################
# USE THIS FOR MULTITHREAD-CPU TRAINING
##$ -q long.q@@stable
##$ -pe smp 7
#THREADS=6


#switch dir in SGE
[ -d "$SGE_O_WORKDIR" ] && cd $SGE_O_WORKDIR
hostname
pwd

########################################################
# SELECT TNET LOCATION
TNET_ROOT='/mnt/matylda5/iveselyk/Tools/TNet/'
#TNET_ROOT='/mnt/matylda5/iveselyk/DEVEL/TNet_PACKAGE/TNet/trunk/'




########################################################
# FEATURES CONFIG 
#
# set LIST and counts to prepare on the fly
LIST='/mnt/matylda5/iveselyk/EXP/AMI_100h/01TJoiner/ihmtrain07.v1_1.crbe.vtln.rand.good_parts.join_ext25_cmncvn.scp'
train_start=0
train_count=199947
cv_start=199947
cv_count=22216
SCP_CV='lists/cv.scp'
SCP_TRAIN='lists/train.scp'

# or set existing SCP_CV SCP_TRAIN lists
#SCP_CV='/mnt/matylda5/iveselyk/DATABASE/TIMIT/lists/fbank23_TJoiner/fbank23_cv_tjoiner15.scp'
#SCP_TRAIN='/mnt/matylda5/iveselyk/DATABASE/TIMIT/lists/fbank23_TJoiner/fbank23_train_tjoiner15.scp'

# FEATURE NORMALIZATION
#STK_CONF='norm.conf'



########################################################
# THIS IS THE SED ARGUMENT THAT WILL CHANGE DISTANT 
# FILE LOCATION TO LOCAL FILE LOCATION!!!!!
COPY_LOCAL=1
SED_ARG_CREATE_LOCAL='s|mnt/scratch05|tmp|g'
###################

# OTHER CONFIG
MLF_CV='/mnt/matylda5/iveselyk/DATABASE/AMIDA_2007/Labels/ihmtrain07.v1_2.phn_states.NOsp.monostates.rand.good_parts.join_ext25.mlf'
MLF_TRAIN='/mnt/matylda5/iveselyk/DATABASE/AMIDA_2007/Labels/ihmtrain07.v1_2.phn_states.NOsp.monostates.rand.good_parts.join_ext25.mlf'
PHONELIST='/mnt/matylda6/grezl/AMIDA_2007/Labels/monostates'


# set input transform parameters
FRM_EXT=5
#DIM_FEA=$(HList -z -h $(head -n 1 $SCP_TRAIN) | grep 'Num Comps' | awk '{ print $3 }')
DIM_FEA=23
DCT_BASE=6
#MMF='' #use this to normalize other than HammDctNorm transforms
#FEATURE_TRANSFORM='' #use this to bypass normalization

# set ANN topology parameters
DIM_IN=$((DIM_FEA*DCT_BASE))
DIM_BN1=80
DIM_BN2=30
DIM_TGT=$(cat $PHONELIST | wc -l)
#automatically compute hidden dimension 
#from total number of parameters
NPARAMS=2000000
DIM_HID=$((NPARAMS/(DIM_IN+6*DIM_BN1+2*DIM_BN2+DIM_TGT)))
#DIM_HID=2460 #or bypass here

# set learn rates for part 1 and part 2
LEARNRATE_P1=1.0
LEARNRATE_P2=1.0
#MOMENTUM=0.5 #only with GPU
#WEIGHTCOST=1e-6 #only with GPU

#BUNCHSIZE=512
CACHESIZE=65000
#RANDOMIZE=TRUE

#MAX_ITER=20
#MIN_ITER=1
#KEEP_LRATE_ITER=0
#END_HALVING_INC=0.1
#START_HALVING_INC=0.5
#HALVING_FACTOR=0.5

# END OF CONFIG
########################################################




########################################################
#I. copy data to local
#(or just split the lists if COPY_LOCAL!=1)
echo "I. copy data to local"
source $TNET_ROOT/tools/train/copy_local.sh
#clean the data at scripts' exit
trap "source $TNET_ROOT/tools/train/delete_local.sh" EXIT


########################################################
#II. run global normalization (corpora wide)
#use only a part of data
echo "II. run global normalization (corpora wide)"
SCP_TNORM=$SCP_TRAIN_LOCAL.head30k
head -n 30000 $SCP_TRAIN_LOCAL > $SCP_TNORM
source $TNET_ROOT/tools/train/tnorm_scheduler.sh


########################################################
#III. run the TNet training, part 1
echo "III. run the TNet training, part 1"
#inintialize the network
NN_INIT="nnet_${DIM_IN}_${DIM_HID}_${DIM_BN1}lin_${DIM_HID}_${DIM_TGT}.init"
if [ ! -e $NN_INIT ]; then
  {
    # more commands here...
    python $TNET_ROOT/tools/init/gen_mlp_init.py --dim=${DIM_IN}:${DIM_HID}:${DIM_BN1}:${DIM_HID}:${DIM_TGT} --gauss --negbias --linBNdim=${DIM_BN1}
    # more commands here...
  } > $NN_INIT 
else
  echo using preinitialized network $NN_INIT
fi 

#run the training
LEARNRATE=$LEARNRATE_P1
source $TNET_ROOT/tools/train/training_scheduler.sh


########################################################
#IV. trim the network, renormalize, insert expansion
echo "IV. trim the network, renormalize, insert expansion"
[ -d weights_part1 ] && rm -r weights_part1
mv weights weights_part1
#trim the network
if [ "1" != "$(find weights_part1/*_final_* | wc -l)" ]; then
  echo Cannot find final network for trimming;
  exit 1
fi
NN_TRIMMED=${NN_INIT%_$DIM_BN1*}_${DIM_BN1}lin
echo Trimming network
cat weights_part1/*_final_* | \
 awk '{ if($0~/<biasedlinearity> .* '$DIM_BN1'/) { exit 0; }  print; }' \
 > $NN_TRIMMED
echo Trimmed network: $NN_TRIMMED
#compose transform
MMF=${FEATURE_TRANSFORM%.transf}_${NN_TRIMMED}
cat ${FEATURE_TRANSFORM} ${NN_TRIMMED} > $MMF
#normalize
unset FEATURE_TRANSFORM
echo Running normalization
source $TNET_ROOT/tools/train/tnorm_scheduler.sh
#add expansion
TMP=${FEATURE_TRANSFORM%.transf}_exp-10_-5_0_5_10.transf
cp $FEATURE_TRANSFORM $TMP
FEATURE_TRANSFORM=$TMP
echo "
<expand> $((5*DIM_BN1)) $DIM_BN1
v 5 -10 -5 0 5 10
" >> $FEATURE_TRANSFORM
#enlarge frmext
FRM_EXT=$((FRM_EXT+10))


########################################################
#V. run the TNet training, part 2
echo "V. run the TNet training, part 2"
#inintialize the network
NN_INIT="nnet_$((5*DIM_BN1))_${DIM_HID}_${DIM_BN2}lin_${DIM_HID}_${DIM_TGT}.init"
if [ ! -e $NN_INIT ]; then
  {
    # more commands here...
    python $TNET_ROOT/tools/init/gen_mlp_init.py --dim=$((5*DIM_BN1)):${DIM_HID}:${DIM_BN2}:${DIM_HID}:${DIM_TGT} --gauss --negbias --linBNdim=${DIM_BN2}
    # more commands here...
  } > $NN_INIT 
else
  echo using preinitialized network $NN_INIT
fi 

#run the training
LEARNRATE=$LEARNRATE_P2
source $TNET_ROOT/tools/train/training_scheduler.sh



########################################################
#VI. trim the network, prepare feature extraction
echo "VI. trim the network, prepare feature extraction"
[ -d weights_part2 ] && rm -r weights_part2
mv weights weights_part2
#trim the network
if [ "1" != "$(find weights_part2/*_final_* | wc -l)" ]; then
  echo Cannot find final network for final trimming;
  exit 1
fi
NN_TRIMMED=${NN_INIT%_$DIM_BN2*}_${DIM_BN2}lin
cat weights_part2/*final* | \
 awk '{ if($0~/<biasedlinearity> .* '$DIM_BN2'/) { exit 0; }  print; }' \
 > $NN_TRIMMED
#compose transform
NN_FINAL=${FEATURE_TRANSFORM}_${NN_TRIMMED}_$(ls weights_part2/*final* | sed 's|.*_||')
cat ${FEATURE_TRANSFORM} ${NN_TRIMMED} > $NN_FINAL

echo 
echo 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% THE END (UC system training finished OK)"
echo "% FINAL BN-feature transform is here:"
echo "% $NN_FINAL"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

