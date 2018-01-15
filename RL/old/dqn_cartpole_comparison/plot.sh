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


# paste <(cut -f2 "$1"/01.tsv) <(cut -f2 "$1"/02.tsv) <(cut -f2 "$1"/03.tsv) <(cut -f2 "$1"/04.tsv) <(cut -f2 "$1"/05.tsv) <(cut -f2 "$1"/06.tsv) <(cut -f2 "$1"/07.tsv) <(cut -f2 "$1"/08.tsv) <(cut -f2 "$1"/09.tsv) <(cut -f2 "$1"/10.tsv) <(cut -f2 "$1"/11.tsv) <(cut -f2 "$1"/12.tsv) <(cut -f2 "$1"/13.tsv) <(cut -f2 "$1"/14.tsv) <(cut -f2 "$1"/15.tsv) <(cut -f2 "$1"/16.tsv) <(cut -f2 "$1"/17.tsv) <(cut -f2 "$1"/18.tsv) <(cut -f2 "$1"/19.tsv) <(cut -f2 "$1"/20.tsv) |
paste <(cut -f2 "$1"/01.tsv) <(cut -f2 "$1"/02.tsv) <(cut -f2 "$1"/03.tsv) <(cut -f2 "$1"/04.tsv) <(cut -f2 "$1"/05.tsv) <(cut -f2 "$1"/06.tsv) <(cut -f2 "$1"/07.tsv) <(cut -f2 "$1"/08.tsv) <(cut -f2 "$1"/09.tsv) |
   mean_row.sh |
   gp.sh 'set yrange[0:200] ; plot "< cat" w l'
