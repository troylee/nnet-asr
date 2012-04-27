
include ../tnet.mk

INCLUDE= -I. -I../ -I../KaldiLib -I../CuBaseLib -I../TNetLib -I$(CUDA_TK_BASE)/include/ 


all : libCuTNet.a

libCuTNet.a : $(OBJ)
	$(AR) ruv $@ $?
	$(RANLIB) $@


%.o : %.cc
	$(CXX) -c $< -o $@ $(CFLAGS) $(CXXFLAGS) $(INCLUDE) 




.PHONY: clean depend

clean :
	rm -f *.o *.a


depend:
	$(CXX) -M $(CXXFLAGS) *.cc $(INCLUDE) > .depend.mk

-include .depend.mk

