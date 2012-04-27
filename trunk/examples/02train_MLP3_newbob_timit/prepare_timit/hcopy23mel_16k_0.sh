#!/bin/bash

echo "SOURCEKIND   = WAVEFORM                                          " >  /tmp/hcopy23mel_$$.cfg
echo "SOURCEFORMAT = NOHEAD                                            " >> /tmp/hcopy23mel_$$.cfg
echo "SOURCERATE   = 625                                               " >> /tmp/hcopy23mel_$$.cfg
echo "BYTEORDER    = VAX                                               " >> /tmp/hcopy23mel_$$.cfg
echo "TARGETFORMAT = HTK                                               " >> /tmp/hcopy23mel_$$.cfg
echo "#TARGETKIND   = MFCC_D_A_0                                       " >> /tmp/hcopy23mel_$$.cfg
echo "TARGETKIND   = FBANK                                             " >> /tmp/hcopy23mel_$$.cfg
echo "                                                                 " >> /tmp/hcopy23mel_$$.cfg
echo "LOFREQ       = 0                                                 " >> /tmp/hcopy23mel_$$.cfg
echo "HIFREQ       = 8000                                              " >> /tmp/hcopy23mel_$$.cfg
echo "NUMCHANS     = 23       # number of critical bands               " >> /tmp/hcopy23mel_$$.cfg
echo "USEPOWER     = T        # using power spectrum                   " >> /tmp/hcopy23mel_$$.cfg
echo "USEHAMMING   = T        # use hamming window on speech frame     " >> /tmp/hcopy23mel_$$.cfg
echo "                                                                 " >> /tmp/hcopy23mel_$$.cfg
echo "PREEMCOEF    = 0        # no preemphase                          " >> /tmp/hcopy23mel_$$.cfg
echo "TARGETRATE   = 100000   # 10 ms frame rate                       " >> /tmp/hcopy23mel_$$.cfg
echo "WINDOWSIZE   = 250000   # 25 ms window                           " >> /tmp/hcopy23mel_$$.cfg
echo "SAVEWITHCRC  = F                                                 " >> /tmp/hcopy23mel_$$.cfg
echo "                                                                 " >> /tmp/hcopy23mel_$$.cfg
echo "#CEPLIFTER   = 22                                                " >> /tmp/hcopy23mel_$$.cfg
echo "NUMCEPS      = 12                                                " >> /tmp/hcopy23mel_$$.cfg

mkdir -p /tmp
HCopy -T 1 -C /tmp/hcopy23mel_$$.cfg -S $1
rm /tmp/hcopy23mel_$$.cfg
