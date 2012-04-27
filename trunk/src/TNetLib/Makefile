
include ../tnet.mk

INCLUDE = -I. -I../KaldiLib -I../STKLib/ 

all: libTNetLib.a

libTNetLib.a: $(OBJ)
	$(AR) ruv $@ $(OBJ) 
	$(RANLIB) $@

%.o : %.cc
	$(CXX)  -o $@  -c $< $(CFLAGS) $(CXXFLAGS) $(INCLUDE)



.PHONY: clean doc depend
clean:
	rm -f *.o *.a

doc:
	doxygen ../../doc/doxyfile_TNetLib 

depend:
	$(CXX) -M $(CXXFLAGS) *.cc $(INCLUDE) > .depend.mk

-include .depend.mk


