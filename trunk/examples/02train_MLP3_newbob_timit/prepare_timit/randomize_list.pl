#!/usr/bin/perl -w
#
# Ussage: randomize_list.pl input output
#

$inlist = $ARGV[0];
$outlist = $ARGV[1];
$seed = $ARGV[2];

if(defined $seed) {
  srand($seed);
}

open (INL,$inlist) or die "cannot open $inlist\n";
open (OUTL,">$outlist");

@list =<INL>;
$num =  $#list;
for $i (0..$num) {
    $a[$i]=rand ($num);
    $b{$a[$i]}=$list[$i];
}

@aa = sort @a;

for $i (0..$num) {
    $sortlist[$i]=$b{$aa[$i]};
}
print OUTL @sortlist;

