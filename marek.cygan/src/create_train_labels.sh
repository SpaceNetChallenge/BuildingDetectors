#TODO check num arguments

if [ $# -ne 2 ]
then
  echo Usage ./create_train_labels.sh traindir testdir
  exit 1
fi

traindir=$1
testdir=$2
echo traindir $traindir
echo testdir $testdir

#seq 1 6940 > train_labels.csv
rm -f train_heatmap.csv
touch train_heatmap.csv
for i in `seq 1 6940`
do
  echo $traindir/3band/3band_AOI_1_RIO_img$i,$traindir/8band/8band_AOI_1_RIO_img$i,NONE,NONE >> train_heatmap.csv
done

rm -f test_heatmap.csv
touch test_heatmap.csv
for i in `cat .indices`
do
  echo $testdir/3band/3band_AOI_2_RIO_img${i},$testdir/8band/8band_AOI_2_RIO_img$i,NONE,NONE >> test_heatmap.csv
done
