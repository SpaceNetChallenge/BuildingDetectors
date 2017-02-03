import sys
import os

print sys.argv
assert len(sys.argv) == 4
d = sys.argv[1]

TRAIN_DATA_DIR = sys.argv[2]+"/" #"competition1/spacenet_TrainData/"
TEST_DATA_DIR = sys.argv[3]+"/" #"competition1/spacenet_TestData/"
train_images_path = TRAIN_DATA_DIR+"3band/3band_AOI_1_RIO_img"
train_images8_path = TRAIN_DATA_DIR+"8band/8band_AOI_1_RIO_img"
test_images_path = TEST_DATA_DIR+"3band/3band_AOI_2_RIO_img"
test_images8_path = TEST_DATA_DIR+"8band/8band_AOI_2_RIO_img"

os.system('ls '+d+'/* > .ff')
t = open('.ff', 'r').readlines()
print len(t)
print t[0], t[-1]

ftrain = open('train_rect.csv', 'w')
ftest = open('test_rect.csv', 'w')
for s in t:
  s = s[:-1]
  print s
  j = len(s)-1
  while s[j] < '0' or s[j] > '9':
    j-= 1
  jj = j+1
  while j >= 0 and s[j] >= '0' and s[j] <= '9':
    j-= 1
  fid = s[j+1:jj]
  print fid
  if s.find('test') != -1:
    print >>ftest, '{},{},NONE,{}'.format(test_images_path+fid,test_images8_path+fid,s)
  else:
    print >>ftrain, '{},{},NONE,{}'.format(train_images_path+fid,train_images8_path+fid,s)

ftrain.close()
ftest.close()
