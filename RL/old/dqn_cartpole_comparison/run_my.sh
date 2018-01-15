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
   # time python my.py --torch-seed=46 --replay-memory-seed=42 --agent-seed=44 --env-seed=439177 --alpha=0.001 --epsilon=0.3 --gamma=0.95 --log-stderr-level=warning --lr=1e-2 --n-batch=32 --n-episodes=200 --n-log-steps=100 --n-middle=50 --n-epsilon-decay=500 --n-steps=200  --n-replay-memory=1000000 --n-start-train=500 --n-target-update-interval=100 --dat-file="$dir/$i.tsv" --q-target-mode=mnih2015 --dqn-mode=doubledqn
   # time python my.py --torch-seed=46 --replay-memory-seed=42 --agent-seed=44 --env-seed=439177 --alpha=0.001 --epsilon=0.3 --gamma=0.95 --log-stderr-level=warning --lr=1e-3 --n-batch=32 --n-episodes=200 --n-log-steps=100 --n-middle=50 --n-epsilon-decay=500 --n-steps=200  --n-replay-memory=1000000 --n-start-train=500 --n-target-update-interval=100 --dat-file="$dir/$i.tsv" --q-target-mode=mnih2015 --dqn-mode=doubledqn
   time python my.py --torch-seed=46 --replay-memory-seed=42 --agent-seed=44 --env-seed=439177 --alpha=0.001 --epsilon=0.3 --gamma=0.95 --log-stderr-level=warning --lr=1e-2 --n-batch=32 --n-episodes=200 --n-log-steps=100 --n-middle=50 --n-epsilon-decay=500 --n-steps=200  --n-replay-memory=1000000 --n-start-train=500 --n-target-update-interval=100 --dat-file="$dir/$i.tsv" --q-target-mode=mnih2015 --dqn-mode=doubledqn
done
