#!/bin/bash
cnt=1
for file in *.xml
do
  echo $file
  cat $file | grep -v image | grep -v md5 > "out$file"
  rm $file
  mv "out$file" $file
  let cnt=cnt+1
done



