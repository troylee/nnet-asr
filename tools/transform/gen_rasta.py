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
(options, args) = parser.parse_args()

if(len(sys.argv) == 1):
    parser.print_help()
    sys.exit()

dimIn = int(options.dimIn)

#RASTA FILTER WEIGHTS
#this is FIR version, 
#impulse response is time reversed (actual sample corresponds to last weight)
filter_coef = [-0.00159062, -0.00169215, -0.00180016, -0.00191506, -0.0020373, -0.00216734, -0.00230568, -0.00245286, -0.00260942, -0.00277598, -0.00295317, -0.00314167, -0.0033422, -0.00355553, -0.00378248, -0.00402392, -0.00428076, -0.004554, -0.00484469, -0.00515392, -0.00548289, -0.00583287, -0.00620518, -0.00660125, -0.00702261, -0.00747086, -0.00794772, -0.00845502, -0.00899471, -0.00956884, -0.0101796, -0.0108294, -0.0115206, -0.012256, -0.0130383, -0.0138705, -0.0147558, -0.0156977, -0.0166997, -0.0177656, -0.0188996, -0.020106, -0.0213893, -0.0227546, -0.024207, -0.0257521, -0.0273959, 0.0772384, 0.13536, 0.144, 0.1]
#length of filter
filter_len=len(filter_coef)

#extend to history only
startFrmExt = filter_len-1
endFrmExt = 0

timeContext = (1+startFrmExt+endFrmExt)

# expand the time context
print '<expand>', dimIn*timeContext, dimIn
print 'v', timeContext
for idx in range(-startFrmExt,endFrmExt+1,1):
  print idx,
print '\n'

# 'transpose' the time windows
print '<transpose>', dimIn*timeContext, dimIn*timeContext
print timeContext, '\n'

# use the filter coefficients
print '<sharedlinearity>', dimIn, dimIn*timeContext
print dimIn
print 'm 1', filter_len
for i in range(filter_len): print filter_coef[i],
print '\nv 1 0.0 '

