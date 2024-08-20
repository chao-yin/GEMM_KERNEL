ARCH = $(shell uname -m)
SYS = $(shell uname -s)

CXX=nvcc
CXXFLAGS = -O3 --ptxas-options=-v -lcublas -arch=sm_60 -rdc=true 
LDFLAGS = -O3 --ptxas-options=-v -lcublas -arch=sm_60 -rdc=true 

exe=./sgemm
objs=$(patsubst %.cu,%.o,$(wildcard *.cu))

deps:=$(join $(addsuffix .deps/,$(dir $(objs))),$(notdir $(objs:.o=.d)))
objs:=$(filter-out sgemm.o,$(objs))

.PHONY: all
all: $(exe)

-include $(deps)

.PHONY: clean
clean:
	rm -f $(objs) $(exe) sgemm.o

$(exe):$(objs) sgemm.o
	@mkdir -p $(dir $(exe))
	$(CXX) -o $(exe) $(objs) sgemm.o $(LDFLAGS) $(LIBS)

%.o: %.cu
	@mkdir -p $(dir $@).deps
	$(CXX) $(CXXFLAGS) -MT $@ -MP -MMD -MF $(dir $@).deps/$(notdir $(@:.o=.d)) -o $@ -c $<
