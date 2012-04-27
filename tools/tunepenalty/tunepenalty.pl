#!/usr/bin/perl

if($ARGV[0] eq "" || $ARGV[1] eq "" || $ARGV[2] eq "" || $ARGV[3] eq "" || $ARGV[4] eq "" || $ARGV[5] eq "" || $ARGV[6] eq "" || $ARGV[7] eq "" || $ARGV[8] eq "")
{
	print "SYNTAX: tunepenalty.pl min max work_directory list mlf phoneme_list lexicon net lm_scale [bin_dir]\n";
	exit(1);
}

$min = $ARGV[0];
$max = $ARGV[1];
$dir = $ARGV[2];
$list = $ARGV[3];
$mlf = $ARGV[4];
$list_phoneme = $ARGV[5];
$lexicon = $ARGV[6];
$net = $ARGV[7];
$lmscale = $ARGV[8];
if(length($ARGV[9])>0) {
  $bin_dir="$ARGV[9]/";
}
$x = ($max + $min) / 2;
$delta = ($max - $min) / 4;
$ins = 100;
$del = 1;
$max_iter=100;
$iter=0;
while($ins / $del > 1.01 || $del / $ins > 1.01)
{
	printf "Trying %3.6f ... \n", $x;
	system("rm -f $dir/cv.mlf");
	$comm = "${bin_dir}SVite --HTK-COMPAT=T -S $list -P HTK -H ${dir}/hmmdefs.stk -i $dir/cv.mlf -l '*' -s $lmscale -w $net -p $x  $lexicon $list_phoneme";
	system($comm);

	system("rm -f $dir/cv.txt");
	$comm = "${bin_dir}HResults -I $mlf $list_phoneme $dir/cv.mlf > $dir/cv.txt";
	system($comm);

	open R, "$dir/cv.txt" or die "Can not open $dir/cv.txt";

	$del = -1;
	$ins = -1;
	while($l = <R>)
	{
		chomp $l;
		if($l =~ /D=(.+), S=(.+), I=(.+),/)
		{
			printf "$1 $2 $3\n";
			$del = $1 + 0.0001;
			$ins = $3 + 0.0001;
		}
	}

	close R;

	last if($ins / $del <= 1.01 && $del / $ins <= 1.01);
	
	if($del == -1 && $ins == -1)
	{
	        printf STDERR "ERROR: tunepenalty.pl - recognition or scoring error!\n\n";
		exit 1;
	}

	if($ins > $del)
	{
		$x -= $delta;
	}
	else
	{
		$x += $delta;
	}
	$delta /= 2;
	if ( $iter >= $max_iter )
	{ 
	    last;
	}
        $iter++;
}
print "Penalty=$x\n";
print "$x\n";
