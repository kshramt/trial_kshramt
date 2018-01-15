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
      echo "${0##*/}" '<dir>'
   } >&2
   exit "${1:-1}"
}


if [[ $# -ne 1 ]]; then
   usage_and_exit
fi


readonly dir="$1"
mkdir -p "$dir"


for i in {01..10}
do
   echo "$i"
   time python ch.py | tee "$dir/$i.tsv"
done
