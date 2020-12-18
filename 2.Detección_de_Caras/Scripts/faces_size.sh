find /media/dadepe/Elements/frames/ -type f -iname "*_face.jpg" -print0 |
  xargs -r0 stat -c '%D:%i %b' | awk '
    !seen[$1]++ {sum += $2}
    END {print sum * 512}'
