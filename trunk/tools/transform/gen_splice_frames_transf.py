#!/usr/local/bin/python -u

# ./gen_hamm_dct.py
# script generateing input transform for NN training with TNet
#
# - context expansion, 
# - critical band transposition,
# - band-wise hamming window
# - DCT transform
#   
# author: Karel Vesely

from math import *


from optparse import OptionParser

import sys

parser = OptionParser()
parser.add_option('--dimIn', dest='dimIn', help='dimension of input features')
parser.add_option('--startFrmExt', dest='startFrmExt', help='frame count of left context')
parser.add_option('--endFrmExt', dest='endFrmExt', help='frame count of right context')
(options, args) = parser.parse_args()

if(len(sys.argv) == 1):
    parser.print_help()
    sys.exit()

dimIn = int(options.dimIn)
startFrmExt = int(options.startFrmExt)
endFrmExt = int(options.endFrmExt)


timeContext = (1+startFrmExt+endFrmExt)

# expand the time context
print '<expand>', dimIn*timeContext, dimIn
print 'v', timeContext
for idx in range(-startFrmExt,endFrmExt+1,1):
  print idx,
print '\n'

