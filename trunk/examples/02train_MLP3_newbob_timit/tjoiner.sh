#!/bin/bash

#$ -o _$JOB_NAME.out
#$ -e _$JOB_NAME.err
#$ -cwd

#$ -q long.q@@stable

if [ $SGE_O_WORKDIR ]; then cd $SGE_O_WORKDIR; fi

TNET_ROOT='../../'

##################################################################
# CONFIG
#
SCP_IN='prepare_timit/workdir/lists/train_fea.scp'
#CONF='norm.conf'

FRM_EXT=15 #select start/end extension of segments in frames

SRC_PATH=$(head -n 1 $SCP_IN); SRC_PATH=${SRC_PATH#*=}; SRC_PATH=${SRC_PATH%/*}
DST_PATH=$SRC_PATH
#On this line we can modify target root location for the joined features
DST_PATH=${DST_PATH%/}${CONF:+_$(basename $CONF .conf)}_tjoiner${FRM_EXT}
#
# 
##################################################################

#append x unitil we get nonexisting directory
while [ -e $DST_PATH ]; do
  DST_PATH=${DST_PATH}x
done

#create output directory
echo "
OUTPUT DIRECTORY:
$DST_PATH

"
mkdir -p $DST_PATH || exit 1

#get the name of the list
SCP_OUT=$(basename $SCP_IN .scp)${CONF:+_$(basename $CONF .conf)}_tjoiner${FRM_EXT}.scp
while [ -e $SCP_OUT ]; do
  SCP_OUT=${SCP_OUT%.scp}x.scp
done

#run the TSegmenter
$TNET_ROOT/src/TJoiner -A -D -V -T 021 \
  -S $SCP_IN \
  ${CONF:+ -C $CONF} \
  -l $DST_PATH \
  --OUTPUT-SCRIPT=$SCP_OUT \
  --START-FRM-EXT=$FRM_EXT \
  --END-FRM-EXT=$FRM_EXT \

if [ $? != 0 ]; then echo 'error occured in tjoiner'>&2; exit 1; fi

echo 'TJoiner finished OK'
