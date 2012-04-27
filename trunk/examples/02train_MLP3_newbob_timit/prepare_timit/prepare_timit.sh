#!/bin/bash


###############################################
##### Set directory with the TIMIT database
in_dir=/mnt/matylda2/data/TIMIT/timit
##### Set directory with the TIMIT database
###############################################



out_dir=$PWD/workdir



mkdir -p $out_dir/data/train
mkdir -p $out_dir/data/test

# convert SPHERE audio files to RAW audio files and asign unique names
for f in `find $in_dir -name "*.wav"` ; do
  file_name=${f##*/}                                                                        # cut off path
  base_name=${file_name%.*}                                                                 # cut off extension
  spk_name=`echo $f | tr '/' ' ' | awk '{print $(NF-1)}'`                                   # timit/test/dr1/faks0/sa1.raw -> faks0
  set_name=`echo $f | tr '/' ' ' | awk '{print $(NF-3)}'`                                   # timit/test/dr1/faks0/sa1.raw -> test 
  echo "$f -> $out_dir/data/${set_name}/${spk_name}_${base_name}.raw"
  sox -t .sph $f -t .raw -w -s -r 16000 $out_dir/data/${set_name}/${spk_name}_${base_name}.raw   # sphere format -> raw format
done

# copy label files and assign unique names
for f in `find $in_dir -name "*.txt" -o -name "*.phn" -o -name "*.wrd"` ; do
  echo $f
  file_name=${f##*/}                                                                        # cut off path
  spk_name=`echo $f | tr '/' ' ' | awk '{print $(NF-1)}'`                                   # timit/test/dr1/faks0/sa1.raw -> faks0
  set_name=`echo $f | tr '/' ' ' | awk '{print $(NF-3)}'`                                   # timit/test/dr1/faks0/sa1.raw -> test 
  echo "$f -> $out_dir/data/${set_name}/${spk_name}_${file_name}"
  cp $f $out_dir/data/${set_name}/${spk_name}_${file_name}
done

# map labels to our reduced set (39 phonemes)
./timit2our39.sh $out_dir/data 

# join all label files to a MLF
mkdir -p $out_dir/mlfs
find $out_dir/data -name "*.lab" > tmp.scp
touch empty.cmd
HLEd -S tmp.scp -i $out_dir/mlfs/ref.mlf -l '*' empty.cmd 
rm -f empty.cmd 
rm -f tmp.scp



# create file lists
export LC_ALL=C
export LANG=en_US

mkdir -p $out_dir/lists

find $out_dir/data/test -name "*.raw" | grep -v "_sa1\|_sa2" | sort > $out_dir/lists/test_raw1.scp
find $out_dir/data/train -name "*.raw" | grep -v "_sa1\|_sa2" | sort > $out_dir/lists/train_cv_raw1.scp

sed -n '1,3296 p' $out_dir/lists/train_cv_raw1.scp > $out_dir/lists/train_raw1.scp
sed -n '3297,3696 p' $out_dir/lists/train_cv_raw1.scp > $out_dir/lists/cv_raw1.scp

./randomize_list.pl $out_dir/lists/train_raw1.scp $out_dir/lists/train_raw.scp 999
./randomize_list.pl $out_dir/lists/cv_raw1.scp $out_dir/lists/cv_raw.scp 999
./randomize_list.pl $out_dir/lists/test_raw1.scp $out_dir/lists/test_raw.scp 999

rm -f $out_dir/lists/train_cv_raw1.scp
rm -f $out_dir/lists/train_raw1.scp
rm -f $out_dir/lists/cv_raw1.scp
rm -f $out_dir/lists/test_raw1.scp

# dictionary
mkdir -p $out_dir/dicts
grep -v "#\|\"\|\." $out_dir/mlfs/ref.mlf | awk '{print $3}' | sort | uniq > $out_dir/dicts/dict






################################
# extract FBANK features by HCopy
mkdir -p $out_dir/features/train
mkdir -p $out_dir/features/test

listL=(train cv test)
for list in ${listL[@]}; do
  # prepare 2-column scp files
  cat $out_dir/lists/${list}_raw.scp | sed -e 's|data|features|' -e 's|.raw$|.fea|' > $out_dir/lists/${list}_fea.scp
  paste $out_dir/lists/${list}_raw.scp $out_dir/lists/${list}_fea.scp > $out_dir/lists/${list}_raw_fea.scp
  # run HCopy feature extraction
  ./hcopy23mel_16k_0.sh $out_dir/lists/${list}_raw_fea.scp
  # remove the scp
  rm $out_dir/lists/${list}_raw_fea.scp
done


################################
# fix the MLF file
# merge unlabelled parts in MLF with next phoneme
awk -f mlf-fix-duration-nosegmenation.awk $out_dir/mlfs/ref.mlf > $out_dir/mlfs/ref.mlf2
mv $out_dir/mlfs/ref.mlf2 $out_dir/mlfs/ref.mlf
# get lengths of feature files
./scp-get-duartion.sh $out_dir/lists/{train,cv,test}_fea.scp > $out_dir/mlfs/duration.lst
./mlf-fix-endduration-nosegmenation.awk $out_dir/mlfs/duration.lst $out_dir/mlfs/ref.mlf >  $out_dir/mlfs/ref.mlf2
mv $out_dir/mlfs/ref.mlf2 $out_dir/mlfs/ref.mlf




