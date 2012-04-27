
include ../tnet.mk 

ifeq ($(BITS64), true)
  BITS=64
  CROSS_COMPILE ="" #x86_64-linux-
else
  BITS=32
endif

ifeq ($(DOUBLEPRECISION), true)
  PRECISION := --enable-double-precision
endif

STKCONFIG := CFLAGS="-m$(BITS) -g -O2 -fPIC" CXXFLAGS="-m$(BITS) -g -O2 -fPIC" CXX="$(CROSS_COMPILE)g++" CC="$(CROSS_COMPILE)g++" AR="$(CROSS_COMPILE)ar" RANLIB="$(CROSS_COMPILE)ranlib" AS="$(CROSS_COMPILE)as" --with-atlas $(PRECISION)



all : libSTKLib.a

libSTKLib.a : libSTKLib.a
	echo $(CC) $(CXX)
	#svn co http://merlin.fit.vutbr.cz/svn/STK/trunk/
	(cd trunk/; ./reconf; ./configure $(STKCONFIG) ; make clean; make SERest; cd ..;)
	cp trunk/src/STKLib/libSTKLib.a .
	cp trunk/src/STKLib/*.h .
	cp trunk/src/STKLib/*.tcc .

clean :
	rm -f libSTKLib.a
