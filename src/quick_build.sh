#make depend && make clean && make #build the CPU tools
make clean
make depend CUDA=true BITS64=true
make CUDA=true BITS64=true #build the CUDA dependent tools
