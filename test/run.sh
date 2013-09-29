#!/bin/bash
# usage: $./run.sh <num_of_threads> 

bin=../amf_d
nt=${1:-16}

wrong_run=0
for i in $(seq 1 10000); do
  result=$($bin bo-KZ2.max $nt 2 2>&1)
  flow=$(echo "$result" | grep FLOW | awk '{print $2}')
  time=$(echo "$result" | grep Wall | awk '{print $3}')
  echo [$wrong_run/$i] $flow $time
  if [[ $flow -ne 202590 ]]; then
    let wrong_run=wrong_run+1
    echo wrong $result
  fi
done
