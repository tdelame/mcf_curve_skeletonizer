####################################################################################################
# 1. COMMANDS
####################################################################################################
AR			      = ar
CXX			      = g++ 
LD 			      = $(CXX)
MAKE				  = make
SHELL		      = bash

####################################################################################################
# 2. GLOBAL OPTIONS TO COMMANDS 
####################################################################################################
ARFLAGS			  = rs 
CXXFLAGS		  = -frounding-math -std=c++14 -fPIC -fopenmp 
LDFLAGS			  = -L/usr/lib -L/usr/local/lib -L$(BINDIR)/$(BUILD_CONF)/lib 
MAKEFLAGS		  = 
SHELLFLAGS    = 

####################################################################################################
# 3. DIRECTORIES
####################################################################################################
BUILD_CONF    = ""
-include $(BINDIR)/build_conf.mk
ifeq ($(BUILD_CONF),"")
	BUILD_CONF  = release
endif
# sources and headers
BINDIR        = bin

# compilation directories
OBJDIR			  = $(BINDIR)/$(BUILD_CONF)/obj
RULESDIR		  = $(BINDIR)/$(BUILD_CONF)/rules
LIBDIR        = $(BINDIR)/$(BUILD_CONF)/lib

####################################################################################################
# 4. LIBRARIES
####################################################################################################
# FLANN
FLANN_INCLUDES    =
FLANN_LDLIBS      = -lflann_cpp -llz4
# OPENMESH
OPENMESH_INCLUDES = -isystem /usr/local/include/OpenMesh
OPENMESH_LDLIBS   = -L/usr/local/lib -L/usr/local/lib/OpenMesh -lOpenMeshCore
# EIGEN
EIGEN_INCLUDES    = -isystem /usr/include/eigen3                    

####################################################################################################
# 5. FLAGS
####################################################################################################
CPPFLAGS      =  
STRIPCMD      = strip --strip-debug --strip-unneeded
LDLIBS        = $(OPENMESH_LDLIBS) $(FLANN_LDLIBS)
INCLUDES      = -I. $(FLANN_INCLUDES) $(OPENMESH_INCLUDES) $(EIGEN_INCLUDES)

ifeq ($(BUILD_CONF), release)
CPPFLAGS     += -UDEBUG -DNDEBUG -DNO_DEBUG -DEIGEN_NO_DEBUG
CXXFLAGS     += -O3 -msse2
else
BUILD_CONF    = debug
CPPFLAGS     += -DDEBUG -O0 
CXXFLAGS     += -ggdb -Wall -Wextra -Wreorder -Wctor-dtor-privacy -Wwrite-strings -fno-inline -fno-inline-functions -fno-inline-small-functions
endif



# COMMAND SHORTCUT
HOSTCOMPILER  = $(CXX) $(CXXFLAGS) -c $(CPPFLAGS) $(INCLUDES)
LINKER        = $(LD)  $(LDFLAGS)     $(CPPFLAGS) $(LDLIBS)
HOSTRECIPER   = $(CXX) $(CXXFLAGS) -M $(CPPFLAGS) $(INCLUDES)
 

####################################################################################################
# 6. PRODUCTS
####################################################################################################
source_names  = $(notdir $(wildcard *.cc))
OBJ				    = $(source_names:%.cc=$(OBJDIR)/%.o)
RULES			    = $(source_names:%.cc=$(RULESDIR)%.d)
BIN			      = mcf_curve_skeletonizer

.SUFFIXES :
.SUFFIXES : .cc .d .h 
.PHONY    : debug release clean help
.SECONDARY: 

all: mcf_curve_skeletonizer

debug:
	@mkdir -p $(BINDIR)
	@echo "BUILD_CONF = debug" > $(BINDIR)/build_conf.mk
	@echo "Build configuration \[debug\] activated and ready
	
release:
	@mkdir -p $(BINDIR)
	@echo "BUILD_CONF = release" > $(BINDIR)/build_conf.mk
	@echo "Build configuration \[release\] activated and ready"
	
$(RULESDIR)/%.d: %.cc
	@echo -e "\033[1;30m[: > host recipe ] \033[0m$$(basename $<)"
	@mkdir -p $(RULESDIR)
	@$(HOSTRECIPER) $< -o $(RULESDIR)/$(*F).temp
	@sed -e 's,\($$*\)\.o[ :]*,\1.o $@ : ,g' \
		< $(RULESDIR)/$(*F).temp \
		> $@;
	@rm $(RULESDIR)/$(*F).temp
-include $(RULES)	

$(OBJDIR)/%.o: %.cc $(RULESDIR)/%.d 
	@echo -e "\033[1;38m[: > host compiling ] \033[0m$$(basename $<)"
	@mkdir -p $(OBJDIR)
	@$(HOSTCOMPILER) -o $@ $< 

mcf_curve_skeletonizer:	$(OBJ)
	@echo -e "\033[1;38m[: > building app ] \033[0m$$(basename $<)"
	@$(LINKER) $(OBJ) -o $@

help:	
	@cat README.md

clean:	
	@echo "In order to avoid removing unexpected files due to bad variable definitions:"
	@echo "  remove $(BINDIR)/ manually"
