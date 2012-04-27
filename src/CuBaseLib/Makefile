
include ../tnet.mk

INCLUDE = -I. -I../ -I../KaldiLib 


CUDA_INCLUDE= -I$(CUDA_TK_BASE)/include
CUDA_BIN=$(CUDA_TK_BASE)/bin


CUSRC=$(wildcard *.cu)
CUOBJ=$(patsubst %.cu, %.o, $(CUSRC))



CUDA_FLAGS = -g -Xcompiler -fPIC --verbose 
ifeq ($(BITS64), true)
  CUDA_FLAGS += --machine 64
  BUT_FORCE_GCC64  = (rm -f $(PWD)/gcc 2>/dev/null; ln -s -f /usr/bin/gcc-4.5 $(PWD)/gcc)
  BUT_UNLINK_GCC64 = unlink $(PWD)/gcc
else
  CUDA_FLAGS += --machine 32
endif

ifeq ($(DOUBLEPRECISION), true)
  CUDA_FLAGS += --gpu-architecture compute_13 --gpu-code sm_13
endif




all : libCuBase.a

libCuBase.a : $(CUOBJ) $(OBJ)
	$(AR) ruv $@ $?
	$(RANLIB) $@


%.o : %.cu
	$(BUT_FORCE_GCC64)
	export PATH=$(PWD):$(CUDA_BIN):$(PATH); $(CUDA_BIN)/nvcc -c $< -o $@ -I. $(CUDA_INCLUDE) $(CUDA_FLAGS)
	$(BUT_UNLINK_GCC64)

%.o : %.cc
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(CUDA_INCLUDE) $(INCLUDE)




.PHONY: clean depend

clean :
	rm -f *.o *.a

depend:
	$(CXX) -M $(CXXFLAGS) *.cc $(INCLUDE) $(CUDA_INCLUDE) > .depend.mk

-include .depend.mk

