#!/bin/bash
# usage: $./run.sh <num_of_threads> 

bin=$(dirname $0)/../amf_d
graph=$(dirname $0)/bo-KZ2.max
nt=${1:-16}
nr=${2:-10000}

wrong_run=0
for i in $(seq 1 $nr); do
  result=$($bin $graph $nt 2 2>&1)
  flow=$(echo "$result" | grep FLOW | awk '{print $2}')
  time=$(echo "$result" | grep Wall | awk '{print $3}')
  echo [$wrong_run/$i] $flow $time
  if [[ $flow -ne 202590 ]]; then
    let wrong_run=wrong_run+1
    echo wrong $result
  fi
done
