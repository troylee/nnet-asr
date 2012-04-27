label_dir=$1

echo "\
ME b  bcl b
ME d  bcl d
ME g  bcl g
ME k  bcl k
ME p  bcl p
ME t  bcl t

ME b  dcl b
ME d  dcl d
ME g  dcl g
ME k  dcl k
ME p  dcl p
ME t  dcl t

ME b  gcl b
ME d  gcl d
ME g  gcl g
ME k  gcl k
ME p  gcl p
ME t  gcl t

ME b  kcl b
ME d  kcl d
ME g  kcl g
ME k  kcl k
ME p  kcl p
ME t  kcl t

ME b  pcl b
ME d  pcl d
ME g  pcl g
ME k  pcl k
ME p  pcl p
ME t  pcl t

ME b  tcl b
ME d  tcl d
ME g  tcl g
ME k  tcl k
ME p  tcl p
ME t  tcl t

RE b  bcl
RE d  dcl
RE g  gcl
RE k  kcl
RE p  pcl
RE t  tcl

RE n   nx
RE aa  ao
RE ah  ax
RE ih  ix
DE q
RE m   em
RE n   en
RE ng  eng
RE sh  zh
RE l   el
RE pau h#
RE pau epi
RE hh  hv
RE uw  ux
RE er  axr
RE ah  ax-h
" > timit2our39.led

for inp in `find $label_dir -name '*.phn' -printf '%p\n'`; do
  echo $inp
  outp=${inp/.phn/.lab}
  HLEd -G TIMIT -n tmp.led timit2our39.led $inp
  cut -f 3 -d " " $outp |  sort | uniq | sed 's/\(.*\)/ME \1 \1 \1/' > tmp.led
  oldsize=""
  while newsize=`wc -c $outp`; [ "$oldsize" != "$newsize" ]; do
    oldsize=$newsize
    HLEd tmp.led $outp
  done
done

rm -f timit2our39.led
rm -f tmp.led
