#
# This makefile contains some global definitions,
# that are used during the build process.
# It is included by all the subridrectory libraries.
#


##############################################################
##### 64-BIT CROSS-COMPILATION #####
CXXFLAGS=
FWDPARAM=
ifeq ($(BITS64), true)
  ##### CHANGE WHEN DIFFERENT 64BIT g++ PREFIX ##### 
  CROSS_COMPILE = "" #x86_64-linux-
  ##### CHANGE WHEN DIFFERENT 64BIT g++ PREFIX ##### 
  CXXFLAGS += -m64
  FWDPARAM += BITS64=true
else
  CXXFLAGS += -m32
endif

# disable cross-compile prefix if CXX not exists
CXX=$(CROSS_COMPILE)g++
CXX2=$(notdir $(shell which $(CXX) 2>/dev/null))
ifneq ("$(CXX)", "$(CXX2)")
  CROSS_COMPILE=
endif

# compilation tools
CC = $(CROSS_COMPILE)g++
CXX = $(CROSS_COMPILE)g++
AR = $(CROSS_COMPILE)ar
RANLIB = $(CROSS_COMPILE)ranlib
AS = $(CROSS_COMPILE)as




##############################################################
##### PATH TO CUDA TOOLKIT #####
#CUDA_TK_BASE=/usr/local/share/cuda-3.2.12
CUDA_TK_BASE=/usr/local/cuda
##### PATH TO CUDA TOOLKIT #####




# compilation args
CXXFLAGS += -g -Wall -O2 -DHAVE_ATLAS -rdynamic
CXXFLAGS += -Wshadow -Wpointer-arith -Wcast-qual -Wcast-align -Wwrite-strings -Wconversion

# enable double-precision
ifeq ($(DOUBLEPRECISION), true)
  CXXFLAGS += -DDOUBLEPRECISION
  FWDPARAM += DOUBLEPRECISION=true
endif


# compile all the source .cc files 
SRC=$(wildcard *.cc)
OBJ=$(patsubst %.cc, %.o, $(SRC))




#########################################################
# CONFIGURATION CHECKS
#

#check that CUDA_TK_BASE is set correctly
ifeq ("$(wildcard $(CUDA_TK_BASE)/bin/nvcc)", "$(CUDA_TK_BASE)/bin/nvcc")
  HAVE_CUDA=true
else 
  ifeq ($(CUDA), true)
    $(error %%% CUDA not found! Incorrect path in CUDA_TK_BASE: $(CUDA_TK_BASE) in 'trunk/src/tnet.mk')
  endif
endif

#
#########################################################


