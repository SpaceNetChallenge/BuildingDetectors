d=$1
echo $d

echo "ls $d/train* > .tmp; cat .tmp | grep -v meta > ftrain"
ls $d/train* > .tmp; cat .tmp | grep -v meta > ftrain

echo "ls $d/val* > .tmp; cat .tmp | grep -v meta > fval"
ls $d/val* > .tmp; cat .tmp | grep -v meta > fval

echo "ls $d/val*_a_a_a > fval2"
ls $d/val*_a_a_a > fval2

echo "ls $d/test*_? > .tmp; cat .tmp | grep -v meta > ftest"
ls $d/test*_? > .tmp; cat .tmp | grep -v meta > ftest
