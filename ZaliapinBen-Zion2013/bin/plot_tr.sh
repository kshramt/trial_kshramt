#!/bin/bash

# set -xv
set -o nounset
set -o errexit
set -o pipefail
set -o noclobber

export IFS=$' \t\n'
export LANG=en_US.UTF-8
umask u=rwx,g=,o=


usage_and_exit(){
   {
      echo "${0##*/}" '<q> < <ijtrm>'
   } >&2
   exit "${1:-1}"
}


if [[ $# -ne 1 ]]; then
   usage_and_exit
fi


"${GNUPLOT:-gnuplot}" -e '
set term png;
set size ratio -1;
q = '"$1"';
plot "< cat" using ($4 - q*$6):($5 - (1 - q)*$6) with dots lc "#cc000000";
'
