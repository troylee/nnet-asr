#!/usr/bin/python

import sys

from optparse import OptionParser

usage='usage: %prog [options] infile [infile2...] outfile'
parser = OptionParser(usage=usage)
parser.add_option('--bndim', dest='bndim', help='dim of bottleneck, the network will be trimmed',default=0)
(options, args) = parser.parse_args()

if len(sys.argv) < 3:
    parser.print_help()
    print '[in|out]file can be "-"'
    sys.exit(1)

bndim=int(options.bndim) #dim of bottleneck
if bndim > 0:
    print 'bndim',bndim

#positional arguments
fninL = args[0:-1] #input files
fnout = args[-1] #output file


#define aux functions
def read_int(file):
    line = file.readline()
    line = line.strip()
    tokL = line.split()
    assert(len(tokL) == 1)
    return int(tokL[0])

def read_vector(file):
    line = file.readline()
    line = line.strip()
    tokL = line.split()
    assert(tokL[0] == 'v')
    while(int(tokL[1]) > len(tokL)-2):
        line = file.readline()
        line = line.strip()
        tokL.extend(line.split())
    assert(int(tokL[1]) == len(tokL)-2)
    #print tokL
    return tokL[2:]

def read_matrix(file):
    line = file.readline()
    #line = line.strip()
    tokL = line.split()
    assert(tokL[0] == 'm')
    mat_size = int(tokL[1]) * int(tokL[2])
    while(mat_size > len(tokL)-3):
        line = file.readline()
        #line = line.strip()
        tokL.extend(line.split())
    assert(mat_size == len(tokL)-3)
    #print tokL
    return (tokL[1],tokL[2],tokL[3:])




#read the network
layers=[]
layersdata=[]
vecsize=0

for fnin in fninL: 
    #open file
    if fnin == '-':
        fin = sys.stdin
    else:
        fin = open(fnin,'r')
    #read file
    while(1):
        line = fin.readline()
        if(len(line) == 0):
            break
        line = line.strip()
        tokL = line.split()
        if(len(tokL) == 0):
            continue
     
        if vecsize==0:
            vecsize=tokL[2]

        if fnout != '-': 
           print tokL

        if(tokL[0] == '<biasedlinearity>'):
            #matrix
            (m_rows,m_cols,m_data) = read_matrix(fin)
            layers.append('<Xform> %s %s' % (m_rows,m_cols))
            layersdata.append(' '.join(m_data))
            #bias
            v_data = read_vector(fin)
            layers.append('<Bias> %d' % len(v_data))
            layersdata.append(' '.join(v_data))

        elif(tokL[0] == '<sharedlinearity>'):
            Ninst = read_int(fin)

            (m_rows,m_cols,m_data) = read_matrix(fin)
            v_data = read_vector(fin)
            
            num_blocks = int(tokL[1]) / int(m_rows)
            assert(num_blocks == Ninst)
            layers.append('<NumBlocks> %d' % num_blocks)

            blocks = ''
            for bl in range(num_blocks):
                blocks += '<Block> %d\n' % (bl+1)
                blocks += '<NumLayers> 2\n'
                blocks += '<Layer> 1\n<XForm> %s %s\n' % (m_rows,m_cols)
                blocks += ' '.join(m_data)
                blocks += '\n'
                blocks += '<Layer> 2\n<Bias> %d\n' % len(v_data)
                blocks += ' '.join(v_data)
                blocks += '\n'

            layersdata.append(blocks)

        elif(tokL[0] == '<blocklinearity>'):
            (m_rows,m_cols,m_data) = read_matrix(fin)
            
            num_blocks = int(tokL[1]) / int(m_rows)
            layers.append('<NumBlocks> %d' % num_blocks)

            blocks = ''
            for bl in range(num_blocks):
                blocks += '<Block> %d\n' % (bl+1)
                blocks += '<XForm> %s %s\n' % (m_rows,m_cols)
                blocks += ' '.join(m_data)
                blocks += '\n'

            layersdata.append(blocks)


        elif(tokL[0] == '<sigmoid>'):
            layers.append('<Sigmoid> %s' % tokL[1])
            layersdata.append('')

        elif(tokL[0] == '<softmax>'):
            layers.append('<Softmax> %s' % tokL[1])
            layersdata.append('')

        elif(tokL[0] == '<blocksoftmax>'):
            v_data = read_vector(fin)

            num_blocks = len(v_data)
            layers.append('<NumBlocks> %d' % num_blocks)

            blocks = ''
            for bl in range(num_blocks):
                blocks += '<Block> %d\n' % (bl+1)
                blocks += '<Softmax> %s\n' % (v_data[bl])
            layersdata.append(blocks)


        elif(tokL[0] == '<expand>'):
            dim_in = int(tokL[2])
            v_data = read_vector(fin)

            #cast str list to int list
            ctxL = []
            for i in range(len(v_data)):
                ctxL.append(int(v_data[i]))

            stack_depth = max(ctxL) - min(ctxL) + 1
            layers.append('<Stacking> %d %d' % (stack_depth,dim_in))
            layersdata.append('')

            layers.append('<Copy> %d %d' % (dim_in*len(ctxL),dim_in*stack_depth))
            selector=''
            for ctx in ctxL:
                selector += '%d:%d ' % (1+(ctx-min(ctxL))*dim_in,(ctx-min(ctxL)+1)*dim_in)
            layersdata.append(selector)


        elif(tokL[0] == '<transpose>'):
            bline = fin.readline()
            ctx_len = int(bline)
            layers.append('<Transpose> %d %d' % (ctx_len,int(tokL[1])/ctx_len))
            layersdata.append('')

        elif(tokL[0] == '<window>'):
            layers.append('<Window> %s' % tokL[1])
            v_data = read_vector(fin)
            layersdata.append(' '.join(v_data))


            
        elif(tokL[0] == '<bias>'):
            layers.append('<Bias> %s' % tokL[1])
            v_data = read_vector(fin)
            layersdata.append(' '.join(v_data))


        elif(tokL[0] == '<log>'):
            layers.append('<Log> %s' % tokL[1])
            layersdata.append('')


        else:
            sys.exit('error cannot parse line: '+line)

        #end parsing at the bottleneck point...
        if(int(tokL[1]) == bndim):
            break

    #close file
    if fnin != '-':
        fin.close()

assert(len(layers) == len(layersdata))
assert(len(layers) > 0)

###########################################
### OUTPUT

#open output file
if fnout == '-':
    fout = sys.stdout
else:
    fout = open(fnout,'w')

#print the network
#fout.write('~o <VecSize> %s\n' % vecsize)
fout.write('~x "NNetsFwdComplete"\n')
fout.write('<NumLayers> '+str(len(layers))+'\n')
for i in range(len(layers)):
    fout.write('<Layer> '+str(i+1)+'\n')
    fout.write(layers[i]+'\n')
    if(len(layersdata[i]) > 0):
        fout.write(layersdata[i]+'\n')

#finally close files
if fnout != '-':
    fout.close()
