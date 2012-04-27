#!/bin/awk -f
BEGIN {
  if (ARGC!=3) {
    print "mlf-fix-duration.awk DURATION MLF_IN"; exit 1;
  }

  dur_in = ARGV[1]
  if (ARGV[2]=="-") {
    f_in = "/dev/stdin";
  } else {
    f_in = ARGV[2];
  }

  #load duration to hash
  while(getline < dur_in) {
    dur[$1] = $2
  }


  print "#!MLF!#"
  while(getline < f_in) {
    if($0~/^#/) { 
      continue
    } else if($0~/^"/) {
      print $0
      key=gensub(/^.*\//,"","") #remove dir
      key=gensub(/\..*$/,"","",key) #remove ext
      duration=dur[key]
      prev=""
    } else if($0!~/^.$/) {
      if(length(prev) >0) {
        print prev
      }
      prev=$0
    } else if($0~/^.$/) {
      if(length(prev) >0) {
        split(prev,arr,/ /)
        if(duration*100000 > arr[2]) {
          col2=duration*100000
        } else {
          col2=arr[2]
        }
        print arr[1]" "col2" "arr[3]
      }
      print $0
    }
  }
}
