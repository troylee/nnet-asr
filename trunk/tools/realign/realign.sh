#!/bin/bash

##$ -N TNET_EXAMPLE
#$ -o _$JOB_NAME.out
#$ -e _$JOB_NAME.err
#$ -cwd

#$ -q long.q@@pco203
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
OutputDir='_realign'

NNET='weights/nnet_368_2105_117_final_iters11_tr80.083_cv71.387'
FEATURE_TRANSFORM='tr_23Tcontext31_Ham_dct16.transf'
FRM_EXT=15
PHONESET='/mnt/matylda5/iveselyk/DATABASE/TIMIT/labels/monophones_39'
#CONF='norm.conf'

SCP_REALIGN='realign.scp'
SOURCE_MLF='/mnt/matylda5/iveselyk/DATABASE/TIMIT/labels/ref.mlf'
TARGET_MLF='realigned.mlf'

POSTERIOR_TMP=1
#
########################################################


if [ ${POSTERIOR_TMP:-0} == 1 ]; then
  POST_REALIGN_DIR=/tmp/iveselyk/realign_$$_test
else
  POST_REALIGN_DIR=$OutputDir/posteriors/realign
fi

mkdir -p $OutputDir/{penalty,scoring}
mkdir -p $POST_REALIGN_DIR




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
# REALIGN THE MLF

#dump posteriors
rm $POST_REALIGN_DIR/* 2>/dev/null
${TNET_ROOT}/src/TFeaCatCu \
  -A \
  ${CONF:+-C $CONF} \
  -l $POST_REALIGN_DIR \
  -y lpost \
  -H $NNET \
  -S $SCP_REALIGN \
  --FEATURETRANSFORM=$FEATURE_TRANSFORM \
  --START-FRM-EXT---=$FRM_EXT \
  --END-FRM-EXT-----=$FRM_EXT \
  --LOGPOSTERIOR----=TRUE

ls $POST_REALIGN_DIR/* > $OutputDir/posteriors_realign.scp

#realign
SVite \
  -A -D -T 1 \
  --htkcompat=TRUE \
  -P HTK \
  -S ${OutputDir}/posteriors_realign.scp \
  -H ${OutputDir}/scoring/hmmdefs.stk \
  -i ${OutputDir}/realigned_raw.mlf \
  -l '*' \
  -a -f -L '*' \
  -I $SOURCE_MLF \
  ${OutputDir}/scoring/dict \
  ${OutputDir}/scoring/hmmlist

#delete test posteriors, we don't need them anymore 
rm -r $POST_REALIGN_DIR

#transform state-alignments to training targets
cat ${OutputDir}/realigned_raw.mlf | sed -e 's|\[|__|' -e 's|\].*||' -e 's|\.rec|.lab|' > ${OutputDir}/realigned.mlf
${TARGET_MLF:+cp ${OutputDir}/realigned.mlf $TARGET_MLF}



