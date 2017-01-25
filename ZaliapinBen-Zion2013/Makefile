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

AWK := gawk
GNUPLOT := gnuplot
PYTHON := python3
CXX := clang++
CXXFLAGS := -Wall -O3 -march=native -std=c++14 -I/opt/local/include -L/opt/local/lib -lGeographic

N_EVENTS := 10000
B := 1
DF := 1.6
Q := 0.5
MC := -100

all_files := $(shell git ls-files)
cpp_files := $(filter %.cpp,$(all_files))

# plot

.PHONY: plot
plot: work/mc~$(MC)/n_events~$(N_EVENTS)/b~$(B)/df~$(DF)/q~$(Q)/hs_1981_2016_comb_K4_A.cat_so_SCSN_v2q.tr.pdf
plot: work/mc~$(MC)/n_events~$(N_EVENTS)/b~$(B)/df~$(DF)/q~$(Q)/hs_1981_2016_comb_K4_A.cat_so_SCSN_v2q.tr_hist.pdf
plot: work/mc~$(MC)/n_events~$(N_EVENTS)/b~$(B)/df~$(DF)/hs_1981_2016_comb_K4_A.cat_so_SCSN_v2q.eta.pdf

work/mc~$(MC)/n_events~$(N_EVENTS)/b~$(B)/df~$(DF)/hs_1981_2016_comb_K4_A.cat_so_SCSN_v2q.eta.pdf: bin/plot_eta.py work/mc~$(MC)/n_events~$(N_EVENTS)/b~$(B)/df~$(DF)/hs_1981_2016_comb_K4_A.cat_so_SCSN_v2q.distance
	mkdir -p $(@D)
	$(PYTHON) $< < $(word 2,$^) >| $@

work/mc~$(MC)/n_events~$(N_EVENTS)/b~$(B)/df~$(DF)/q~$(Q)/hs_1981_2016_comb_K4_A.cat_so_SCSN_v2q.tr_hist.pdf: bin/plot_tr_hist.py work/mc~$(MC)/n_events~$(N_EVENTS)/b~$(B)/df~$(DF)/hs_1981_2016_comb_K4_A.cat_so_SCSN_v2q.distance
	mkdir -p $(@D)
	$(PYTHON) $< $(Q) < $(word 2,$^) >| $@

work/mc~$(MC)/n_events~$(N_EVENTS)/b~$(B)/df~$(DF)/q~$(Q)/hs_1981_2016_comb_K4_A.cat_so_SCSN_v2q.tr.pdf: bin/plot_tr.sh work/mc~$(MC)/n_events~$(N_EVENTS)/b~$(B)/df~$(DF)/hs_1981_2016_comb_K4_A.cat_so_SCSN_v2q.distance
	mkdir -p $(@D)
	GNUPLOT=$(GNUPLOT) $< $(Q) < $(word 2,$^) >| $@

.PHONY: diff
diff: work/mc~$(MC)/n_events~$(N_EVENTS)/b~$(B)/df~$(DF)/hs_1981_2016_comb_K4_A.cat_so_SCSN_v2q.distance work/mc~$(MC)/n_events~$(N_EVENTS)/b~$(B)/df~$(DF)/hs_1981_2016_comb_K4_A.cat_so_SCSN_v2q_ref.distance
	diff <(sort -k1 -g $(word 1,$^)) <(sort -k1 -g $(word 2,$^))

%.distance: zaliapin_ben_zion_2013.exe %.in
	mkdir -p $(@D)
	time $(abspath $<) < $(word 2,$^) >| $@

%_ref.distance: zaliapin_ben_zion_2013_ref.exe %.in
	mkdir -p $(@D)
	time $(abspath $<) < $(word 2,$^) >| $@

work/mc~$(MC)/n_events~$(N_EVENTS)/b~$(B)/df~$(DF)/hs_1981_2016_comb_K4_A.cat_so_SCSN_v2q.in: work/mc~$(MC)/hs_1981_2016_comb_K4_A.cat_so_SCSN_v2q.tmyx
	mkdir -p $(@D)
	{
	   echo $(B) $(DF)
	   tail -n$(N_EVENTS) $<
	} >| $@

work/mc~$(MC)/hs_1981_2016_comb_K4_A.cat_so_SCSN_v2q.tmyx: work/hs_1981_2016_comb_K4_A.cat_so_SCSN_v2q.tmyx
	mkdir -p $(@D)
	$(AWK) '$$2 >= $(MC)' $< >| $@

work/hs_1981_2016_comb_K4_A.cat_so_SCSN_v2q.tmyx: bin/scec.py cache/hs_1981_2016_comb_K4_A.cat_so_SCSN_v2q
	mkdir -p $(@D)
	$(PYTHON) $< < $(word 2,$^) >| $@

# check

.PHONY: check
check: test/syn.done
test/%.done: bin/allclose.py test/%.out test/%.expect
	mkdir -p $(@D)
	$(PYTHON) $^
	touch $@

test/%.out: zaliapin_ben_zion_2013.exe test/%.in
	mkdir -p $(@D)
	$(abspath $<) < $(word 2,$^) >| $@

# data

cache/hs_1981_2016_comb_K4_A.cat_so_SCSN_v2q:
	mkdir -p $(@D)
	cd $(@D)
	wget http://scedc.caltech.edu/ftp/catalogs/hauksson/Socal_DD/$(@F)

# CXX

all: zaliapin_ben_zion_2013.exe
%.exe: work/build/%.exe
	cp -f $< $@

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
