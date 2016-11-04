# Configurations
.SUFFIXES:
.DELETE_ON_ERROR:
.SECONDARY:
.ONESHELL:
export SHELL := /bin/bash
export SHELLOPTS := pipefail:errexit:nounset:noclobber

# Tasks
.PHONY: all
.DEFAULT_GOAL := all

# Specific to this project

GNUPLOT := gnuplot
PYTHON := python3
CXX := clang++
CXXFLAGS := -Wall -O3 -march=native -std=c++14 -I/opt/local/include -L/opt/local/lib -lGeographic

N_EVENTS := 10000
B := 1
DF := 1.6
Q := 0.5

all_files := $(shell git ls-files)
cpp_files := $(filter %.cpp,$(all_files))

# run

all: work/n_events~$(N_EVENTS)/b~$(B)/df~$(DF)/q~$(Q)/hs_1981_2016_comb_K4_A.cat_so_SCSN_v2q.tr.pdf

work/n_events~$(N_EVENTS)/b~$(B)/df~$(DF)/q~$(Q)/hs_1981_2016_comb_K4_A.cat_so_SCSN_v2q.tr.pdf: bin/plot_tr.sh work/n_events~$(N_EVENTS)/b~$(B)/df~$(DF)/hs_1981_2016_comb_K4_A.cat_so_SCSN_v2q.distance
	mkdir -p $(@D)
	GNUPLOT=$(GNUPLOT) $< $(Q) < $(word 2,$^) >| $@

work/n_events~$(N_EVENTS)/b~$(B)/df~$(DF)/hs_1981_2016_comb_K4_A.cat_so_SCSN_v2q.distance: work/build/zaliapin_ben_zion_2013.exe work/n_events~$(N_EVENTS)/b~$(B)/df~$(DF)/hs_1981_2016_comb_K4_A.cat_so_SCSN_v2q.in
	mkdir -p $(@D)
	$< < $(word 2,$^) >| $@

work/n_events~$(N_EVENTS)/b~$(B)/df~$(DF)/hs_1981_2016_comb_K4_A.cat_so_SCSN_v2q.in: work/n_events~$(N_EVENTS)/hs_1981_2016_comb_K4_A.cat_so_SCSN_v2q.tmyx
	mkdir -p $(@D)
	{
	   echo $(B) $(DF)
	   cat $<
	} >| $@

work/n_events~$(N_EVENTS)/hs_1981_2016_comb_K4_A.cat_so_SCSN_v2q.tmyx: zaliapin_ben_zion_2013.py work/n_events~$(N_EVENTS)/hs_1981_2016_comb_K4_A.cat_so_SCSN_v2q
	mkdir -p $(@D)
	$(PYTHON) $< < $(word 2,$^) >| $@

# data

work/n_events~$(N_EVENTS)/hs_1981_2016_comb_K4_A.cat_so_SCSN_v2q: cache/hs_1981_2016_comb_K4_A.cat_so_SCSN_v2q
	mkdir -p $(@D)
	tail -n$(N_EVENTS) $< >| $@

cache/hs_1981_2016_comb_K4_A.cat_so_SCSN_v2q:
	mkdir -p $(@D)
	cd $(@D)
	wget http://scedc.caltech.edu/ftp/catalogs/hauksson/Socal_DD/$(@F)

# CXX

%.exe: %.o
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $@ $^


work/build/%.o:
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c -o $@ $<


work/build/%.d: %.cpp
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -MM -MT $(@:%.d=%.o) -MF $@ $^


-include $(cpp_files:%.cpp=work/build/%.d)
