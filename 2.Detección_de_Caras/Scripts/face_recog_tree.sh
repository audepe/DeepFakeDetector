#!/bin/bash

total_dirs=$(ls -l /media/dadepe/Elements/frames/fake/ | wc -l)
count=1
for d in /media/dadepe/Elements/frames/fake/* ; do
    echo "Procesados $count de $total_dirs"
    face_detection "$d" >> ./fake_face_locations
    let count=count+1
done