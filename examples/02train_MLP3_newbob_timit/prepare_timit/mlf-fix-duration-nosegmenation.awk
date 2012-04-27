#!/bin/awk -f
BEGIN {
  if (ARGC!=2) {
    print "mlf-fix-duration.awk MLF_IN"; exit 1;
  }
  if (ARGV[1]=="-") {
    f_in = "/dev/stdin";
  } else {
    f_in = ARGV[1];
  }

  print "#!MLF!#"
  while(getline < f_in) {
    if($0~/^#/) { 
      continue
    } else if($0~/^"/) {
      pos=0
      print $0
    } else if($0!~/^.$/) {
      endpos = int($2/100000 +0.5)
      print pos"00000 "endpos"00000 "$3
      pos=endpos
    } else if($0~/^.$/) {
      print $0
    }
  }
}
