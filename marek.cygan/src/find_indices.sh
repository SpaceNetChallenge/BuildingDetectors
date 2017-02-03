testdir=$1
echo testdir $testdir
rm -f .indices
touch .indices
for i in `seq 0 5000`
do
  if [ -f $testdir/3band/3band_AOI_2_RIO_img${i}.tif ]
  then
    echo $i >> .indices
  fi
done
