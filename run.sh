#!/bin/bash

if [ $1 == 1 ]
then
  echo "Running Q1"
  if [ $5 == "a" ]
  then
    echo "DOING part a"
    python Q1/1a.py --train $2 --test $3 --val $4 --type 0
    python Q1/1a.py --train $2 --test $3 --val $4 --type 1
  elif [ $5 == "b" ]
  then
    echo "DOING part b"
    python Q1/1a.py --train $2 --test $3 --val $4 --type 0
    python Q1/1a.py --train $2 --test $3 --val $4 --type 1
  elif [ $5 == "c" ]
  then
    echo "DOING part c"
    python Q1/1c.py  --train $2 --test $3 --val $4
  elif [ $5 == "d" ]
  then
    echo "DOING part d"
    python Q1/1d.py --train $2 --test $3 --val $4 --type n_estimators
    python Q1/1d.py --train $2 --test $3 --val $4 --type max_features
    python Q1/1d.py --train $2 --test $3 --val $4 --type min_samples_split
  else
    echo "Wrong part entered."
  fi
elif [ $1 == 2 ]
then
  echo "Running Q2"
  if [ $4 == "a" ]
  then
    echo "DOING part a"
    python Q2/2a.py --train $2 --test $3 --adaptive 0 --activation sigmoid -q a -l1 10 -l2 0 -k 30 --delta 1e-7
  elif [ $4 == "b" ]
  then
    echo "DOING part b"
    python Q2/2a.py --train $2 --test $3 --adaptive 0 --activation sigmoid -q b -l1 10 -l2 0 -k 30 --delta 1e-7
  elif [ $4 == "c" ]
  then
    echo "DOING part c"
    python Q2/2a.py --train $2 --test $3 --adaptive 0 --activation sigmoid -q c -l1 10 -l2 0 -k 30 --delta 1e-7
  elif [ $4 == "d" ]
  then
    echo "DOING part d"
    python Q2/2a.py --train $2 --test $3 --adaptive 1 --activation sigmoid -q d -l1 10 -l2 0 -k 30 --delta 1e-7
  elif [ $4 == "e" ]
  then
    echo "DOING part e"
    python Q2/2a.py --train $2 --test $3 --adaptive 1 --activation relu -q e -l1 100 -l2 100 -k 50 --delta 4e-8
    python Q2/2a.py --train $2 --test $3 --adaptive 1 --activation sigmoid -q e -l1 100 -l2 100 -k 50 --delta 4e-8
  elif [ $4 == "f" ]
  then
    echo "DOING part f"
    python Q2/2f.py --train $2 --test $3
  else
    echo "Wrong part entered."
  fi
else
  echo "Invalid question number"
fi
