#!/bin/bash

##$ -N TNET_EXAMPLE
#$ -o _$JOB_NAME.out
#$ -e _$JOB_NAME.err
#$ -cwd

#$ -q long.q@@pco203,long.q@pcspeech-gpu
#$ -l gpu=1

if [ $SGE_O_WORKDIR ]; then cd $SGE_O_WORKDIR; fi

hostname

########################################################
# SELECT TNET LOCATION
#TNET_ROOT='/mnt/matylda5/iveselyk/Tools/TNet/'
TNET_ROOT='/mnt/matylda5/iveselyk/DEVEL/TNet_PACKAGE/TNet/trunk/'



########################################################
# CONFIG
#
OutputDir='_tpenalty'

NNET='/mnt/matylda5/iveselyk/EXP/AMI_10h_task_best_setup__init_lrate_mmt_wc/12_weightcost_not_in_momentum_1M_nnet_bottleneck/lrate1.0_mmt0.0_wc0.0-test-sigmoid-thr0.001/weights/nnet_276_2036_80_2036_135_final_iters13_accu48.8488'
FEATURE_TRANSFORM='/mnt/matylda5/iveselyk/EXP/AMI_10h_task_best_setup__init_lrate_mmt_wc/12_weightcost_not_in_momentum_1M_nnet_bottleneck/lrate1.0_mmt0.0_wc0.0-test-sigmoid-normal/tr_Tcontext21_Ham_23dct12_allBand_cmncvn'
FRM_EXT=10
PHONESET='/mnt/matylda6/grezl/AMIDA_2007/Labels/monophones'
#CONF=''

SCP_CV='/mnt/matylda5/iveselyk/EXP/AMI_10h_task_best_setup__init_lrate_mmt_wc/12_weightcost_not_in_momentum_1M_nnet_bottleneck/lrate1.0_mmt0.0_wc0.0-test-sigmoid-normal/lists/dev_cmncvn.scp'
SCP_TEST='/mnt/matylda5/iveselyk/EXP/AMI_10h_task_best_setup__init_lrate_mmt_wc/12_weightcost_not_in_momentum_1M_nnet_bottleneck/lrate1.0_mmt0.0_wc0.0-test-sigmoid-normal/lists/dev_cmncvn.scp'
MLF_REF='/mnt/matylda5/iveselyk/DATABASE/AMIDA_2007/Labels/ihmtrain07.v1_2.phn_states.NOsp.monophones_join.mlf'

LMSCALE=0.0
WIPRANGE=10

TUNE_SAME_DEL_INS=1
TUNE_ACC=1
POSTERIOR_TMP=1
#
########################################################


if [ ${POSTERIOR_TMP:-0} == 1 ]; then
  POST_CV_DIR=/tmp/iveselyk/tpenalty_$$_cv
  POST_TEST_DIR=/tmp/iveselyk/tpenalty_$$_test
else
  POST_CV_DIR=$OutputDir/posteriors/cv
  POST_TEST_DIR=$OutputDir/posteriors/test
fi

mkdir -p $OutputDir/{penalty,scoring}
mkdir -p {$POST_CV_DIR,$POST_TEST_DIR}

##########################################
#prepare decoder

# create HMM model for STK decoder with posterior input
cp $PHONESET ${OutputDir}/scoring/hmmlist

# create dictonary
awk '{print $1 " " $1}' ${OutputDir}/scoring/hmmlist > ${OutputDir}/scoring/dict

#create recognition net
HBuild ${OutputDir}/scoring/hmmlist ${OutputDir}/scoring/monophones_lnet.hvite

# generate hmm model with obscoef input
awk '{print $1, NF - 1;}' ${OutputDir}/scoring/hmmlist > ${OutputDir}/scoring/hmmlist.count
${TNET_ROOT}/tools/tunepenalty/create_like_hmms.pl ${OutputDir}/scoring/hmmlist.count > ${OutputDir}/scoring/hmmdefs.stk
cp ${OutputDir}/scoring/hmmdefs.stk ${OutputDir}/penalty/hmmdefs.stk




##########################################
# TUNE THE WIP ON HELD-OUT DATA

#dump posteriors
rm $POST_CV_DIR/* 2>/dev/null
${TNET_ROOT}/src/TFeaCatCu \
  -A \
  ${CONF:+-C $CONF} \
  -l $POST_CV_DIR \
  -y lpost \
  -H $NNET \
  -S $SCP_CV \
  --FEATURETRANSFORM=$FEATURE_TRANSFORM \
  --START-FRM-EXT---=$FRM_EXT \
  --END-FRM-EXT-----=$FRM_EXT \
  --LOGPOSTERIOR----=TRUE

ls $POST_CV_DIR/* > $OutputDir/posteriors_cv.scp

#run tuning, same no. of insertions/deletions
if [ ${TUNE_SAME_DEL_INS:-0} == 1 ]; then
${TNET_ROOT}/tools/tunepenalty/tunepenalty.pl \
 -${WIPRANGE} ${WIPRANGE} \
 $OutputDir/penalty \
 ${OutputDir}/posteriors_cv.scp \
 $MLF_REF \
 $OutputDir/scoring/hmmlist \
 $OutputDir/scoring/dict \
 $OutputDir/scoring/monophones_lnet.hvite \
 $LMSCALE | \
 tee ${OutputDir}/tunepenalty.log

 wip=$(cat ${OutputDir}/tunepenalty.log | tail -n 1)
fi

#run tuning, best accuracy
if [ ${TUNE_ACC:-0} == 1 ]; then
${TNET_ROOT}/tools/tunepenalty/tunepenalty_acc.py \
 -${WIPRANGE} ${WIPRANGE} \
 $OutputDir/penalty \
 ${OutputDir}/posteriors_cv.scp \
 $MLF_REF \
 $OutputDir/scoring/hmmlist \
 $OutputDir/scoring/dict \
 $OutputDir/scoring/monophones_lnet.hvite \
 $LMSCALE | \
 tee ${OutputDir}/tunepenalty_acc.log

 wip=$(cat ${OutputDir}/tunepenalty_acc.log | tail -n 1)
fi

#detele cv posteriors, we don't need them anymore 
rm -r $POST_CV_DIR


##########################################
# DECODE TEST DATA
if [ ! -f ${SCP_TEST:-"/dummy"} ]; then exit; fi

#dump posteriors
rm $POST_TEST_DIR/* 2>/dev/null
${TNET_ROOT}/src/TFeaCatCu \
  -A \
  ${CONF:+-C $CONF} \
  -l $POST_TEST_DIR \
  -y lpost \
  -H $NNET \
  -S $SCP_TEST \
  --FEATURETRANSFORM=$FEATURE_TRANSFORM \
  --START-FRM-EXT---=$FRM_EXT \
  --END-FRM-EXT-----=$FRM_EXT \
  --LOGPOSTERIOR----=TRUE

ls $POST_TEST_DIR/* > $OutputDir/posteriors_test.scp

#decode
SVite \
  -A -D \
  --htkcompat=TRUE \
  -P HTK \
  -S ${OutputDir}/posteriors_test.scp \
  -H ${OutputDir}/scoring/hmmdefs.stk \
  -i ${OutputDir}/hyp_test.mlf \
  -l '*' \
  -s $LMSCALE \
  -w ${OutputDir}/scoring/monophones_lnet.hvite \
  -p ${wip:-0.0} \
  ${OutputDir}/scoring/dict \
  ${OutputDir}/scoring/hmmlist

#score
HResults -I $MLF_REF /dev/null ${OutputDir}/hyp_test.mlf | \
  tee ${OutputDir}/restuts.txt

#detele test posteriors, we don't need them anymore 
rm -r $POST_TEST_DIR
