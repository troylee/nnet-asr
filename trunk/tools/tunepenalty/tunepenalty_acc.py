#!/usr/bin/python

import os, sys, re

import findmax


# arguments
if(not len(sys.argv) in range(10,11+1)):
    print 'SYNTAX: tunepenalty_acc.py min max work_directory list mlf phoneme_list lexicon net lm_scale [bin_dir]'
    sys.exit(1)

min = float(sys.argv[1])
max = float(sys.argv[2])
dir = sys.argv[3]
list = sys.argv[4]
mlf = sys.argv[5]
list_phoneme = sys.argv[6]
lexicon = sys.argv[7]
net = sys.argv[8]
lmscale = float(sys.argv[9])
try:
    if(len(sys.argv[10]) > 0):
        bin_dir = sys.argv[10]+'/'
    else:
        bin_dir = ''
except:
    bin_dir = '' #use system-wide tools

# function that we want to maximize
def tryPenaltyValue(penalty):
    print 'Trying %3.6f ... ' % penalty ,
    os.system('rm -f %s/cv.mlf' % dir)
    
    comm = bin_dir+'SVite --htkcompat=TRUE -P HTK -S %s' % list \
         + ' -H %s/hmmdefs.stk -i %s/cv.mlf -l \'*\'' % (dir,dir) \
         + ' -s %g -w %s -p %g %s %s' % (lmscale,net,penalty,lexicon,list_phoneme)

    if 0 != os.system(comm):
        raise SViteError

    comm = bin_dir+'HResults -I %s %s %s/cv.mlf' % (mlf,list_phoneme,dir)

    pipe = os.popen(comm,'r')
    for line in pipe:
        m = re.search('Acc=(.*) \[',line)
        if (None != m):
            print m.group(1), '\t', line.rstrip()
            pipe.close()
            return float(m.group(1))

    
    if None != pipe.close():
        raise HResultsError
    print 'Error: Could not find accuracy in the HResults output!!!'
    raise NoAccFoundError 


# run the maximization
penalty = findmax.findMax(tryPenaltyValue,min,max)
print 'Guessing %g' % penalty
tryPenaltyValue(penalty)
print 'Penalty=%g' % penalty
print '%g' % penalty




