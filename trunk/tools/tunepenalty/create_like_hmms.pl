#!/usr/bin/perl

if($ARGV[0] eq "")
{
	print "syntax: create_like_hmms.pl list [nstates]\n";
	exit(1);
}

$vct_size = 0;
@phn_list = ();
%phns = ();
$def_nstates=3;

if($ARGV[1] ne "")
{
  $def_nstates = $ARGV[1];
}

open L, $ARGV[0] or die "Can not open $ARGV[0]";
while($l = <L>)
{
  chomp $l;
  ($phn, $n) = split /\s+/, $l;
  $n = $def_nstates if($n == "");
  $phns{$phn} = $n;
  $vct_size += $n;
  push @phn_list, $phn;
}
close L;


print "~o <VECSIZE> $vct_size\n";
$idx = 1;

foreach $phn (@phn_list)
{
	print "~h \"$phn\"\n";
	print "<BEGINHMM>\n";
	$nstates = $phns{$phn} + 2;
	print "<NUMSTATES> $nstates\n";
	for($i = 2; $i < $nstates; $i++)
	{
	    print "<STATE> $i <OBSCOEF> $idx\n";
	    $idx++;
	}
	print "<TRANSP> $nstates\n";
	for($j = 0; $j < $nstates; $j++)
	{
	    if($j == 1)
	    {
	      print " 1.0";
            }
	    else
	    {
 	      print " 0.0";
	    }
        }
	print "\n";
	for($i = 1; $i < $nstates - 1; $i++)
	{
	    for($j = 0; $j < $nstates; $j++)
	    {
		if($j == $i || $j == $i + 1)
		{
		    print " 0.5";
        	}
		else
		{
		    print " 0.0";
        	}
	    }
	    print "\n";	    
	}
	for($j = 0; $j < $nstates; $j++)
	{
 	    print " 0.0";
        }
	print "\n";
	
	print "<ENDHMM>\n";
}
close L;
