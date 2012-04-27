
include tnet.mk


##### Includes
INCLUDE = -IKaldiLib -ITNetLib -ISTKLib 
INCLUDE += -ICuBaseLib -ICuTNetLib
INCLUDE += -I$(CUDA_TK_BASE)/include

CXXFLAGS += $(INCLUDE)

##### Link TNetLib & STKLib
LDFLAGS =   -LTNetLib -lTNetLib
LDFLAGS +=   -LSTKLib -lSTKLib
ifeq ($(CUDA), true)
  LDFLAGS += -LCuTNetLib -lCuTNet
  LDFLAGS += -LCuBaseLib -lCuBase
endif
LDFLAGS +=   -LKaldiLib -lKaldiLib


##### Link with GotoBLAS
ifeq ($(BITS64), true) 
  LDFLAGS += -LGotoBLASLib -lgoto2_64 -lgfortran
else
  LDFLAGS += -LGotoBLASLib -lgoto2 -lgfortran
endif
LDFLAGS += -Wl,-rpath,$(PWD)/GotoBLASLib

##### Link with CUBLAS
ifeq ($(CUDA), true)
  ifeq ($(BITS64), true)
    LDFLAGS += -L$(CUDA_TK_BASE)/lib64 -Wl,-rpath,$(CUDA_TK_BASE)/lib64
  else
    LDFLAGS += -L$(CUDA_TK_BASE)/lib -Wl,-rpath,$(CUDA_TK_BASE)/lib
  endif
  LDFLAGS += -lcublas -lcudart -lcuda 
endif

LDFLAGS +=   -pthread 

################################################################################
# Target programs 
################################################################################
BINS := TNet TNorm TFeaCat TSegmenter TJoiner
CUBINS := TNetCu TNormCu TFeaCatCu TRbmCu TMpeCu TMmiCu TRecurrentCu
ifeq ($(CUDA), true)
  BINS += $(CUBINS)
endif

all : $(BINS) 

$(BINS): lib

ifeq ($(CUDA), true)
$(BINS): culib
endif


###############################################################################
# program specific rules
###############################################################################

#TNet : TNet.o
#	$(CXX)  -o $@  $< $(CXXFLAGS) $(INCLUDE) $(LDFLAGS)
#
#TNetCu : TNetCu.o
#	$(CXX)  -o $@  $< $(CXXFLAGS) $(INCLUDE) $(LDFLAGS)
#
#TNormCu : TNormCu.o
#	$(CXX)  -o $@  $< $(CXXFLAGS) $(INCLUDE) $(LDFLAGS)
#
#TFeaCat : TFeaCat.o
#	$(CXX)  -o $@  $< $(CXXFLAGS) $(INCLUDE) $(LDFLAGS)
#
#TFeaCatCu : TFeaCatCu.o
#	$(CXX)  -o $@  $< $(CXXFLAGS) $(INCLUDE) $(LDFLAGS)
#
#TRbmCu : TRbmCu.o
#	$(CXX)  -o $@  $< $(CXXFLAGS) $(INCLUDE) $(LDFLAGS)
#
#TMpeCu : TMpeCu.o
#	$(CXX)  -o $@  $< $(CXXFLAGS) $(INCLUDE) $(LDFLAGS)
#
#TRecurrentCu : TRecurrentCu.o
#	$(CXX)  -o $@  $< $(CXXFLAGS) $(INCLUDE) $(LDFLAGS)
#
#TSegmenter : TSegmenter.o
#	$(CXX)  -o $@  $< $(CXXFLAGS) $(INCLUDE) $(LDFLAGS)
#
#TJoiner : TJoiner.o
#	$(CXX)  -o $@  $< $(CXXFLAGS) $(INCLUDE) $(LDFLAGS)


###############################################################################
# program compliling implicit rule
###############################################################################
% : %.o
	$(CXX)  -o $@  $< $(CXXFLAGS) $(INCLUDE) $(LDFLAGS)

#% : %.cc
#	$(CXX)  -o $@  $< $(CXXFLAGS) $(INCLUDE) $(LDFLAGS)
 
###############################################################################
# module compliling implicit rule
###############################################################################
%.o : %.cc lib
	$(CXX)  -o $@  -c $< $(CFLAGS) $(CXXFLAGS) $(INCLUDE)



###############################################################################
.PHONY: lib culib stklib clean doc depend

lib: stklib
	@cd KaldiLib && make $(FWDPARAM)
	@cd TNetLib && make $(FWDPARAM)

culib:
	@cd CuBaseLib && make $(FWDPARAM)
	@cd CuTNetLib && make $(FWDPARAM)
	
stklib:
	@cd STKLib && make $(FWDPARAM)

clean:
	rm -f *.o $(BINS) $(CUBINS)
	@cd STKLib && make clean
	@cd KaldiLib && make clean
	@cd TNetLib && make clean
	@cd CuBaseLib && make clean
	@cd CuTNetLib && make clean

doc:
	doxygen ../doc/doxyfile_TNet

depend: stklib
	$(CXX) -M $(CXXFLAGS) *.cc $(INCLUDE) > .depend.mk
	@cd KaldiLib && make depend
	@cd TNetLib && make depend
	@cd CuBaseLib && make depend
	@cd CuTNetLib && make depend

-include .depend.mk


