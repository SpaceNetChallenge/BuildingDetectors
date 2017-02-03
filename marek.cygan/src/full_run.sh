if [ $# -ne 3 ]
then
  echo Usage ./create_train_labels.sh traindir testdir outputfile
  exit 1
fi

traindir=$1/
testdir=$2/
output=$3
echo traindir $traindir
echo testdir $testdir
echo outputfile $output

./find_indices.sh $testdir
./create_train_labels.sh $traindir $testdir

rm heatmaps/*
./vis --train-path $traindir --heatmap 400 0.01 --angles 1 --centers 1 --pieces 50

rm predictions/*
rm -rf stored_predictions5
python -u main.py --rectangles 144 --pieces 50 --batch_size 8 --lr 0.0001 --epoch_decay 1.0 --dilarch 2 --num_epochs 80 --mode heatmap --final_bn 1 --restore_path stored_models_heat/models_heat/epoch_13_continue.ckpt --predict 1 |& tee log1

mv predictions stored_predictions5
mkdir predictions

python create_train_rect.py stored_predictions5 $traindir $testdir
python -u main.py --rectangles 16 --pieces 50 --batch_size 16 --lr 0.0002 --epoch_decay 1.0 --mode rectangles --num_epochs 100 --arch_multiplier 5 --mse_mult 10 --restore_path stored_models_rect/submit16/epoch32.ckpt --predict 1 --augment 1 |& tee log2

./prep_f.sh predictions/
./vis --test-path $testdir --train-path $traindir --nms ftest 0.2 -0.997 --centers 1 --pieces 50 --angles 1 --shift
cp submission.csv $output
