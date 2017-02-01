input_dir=`realpath "$1"`
output_dir=`dirname "$3"`
output_dir=`realpath "$output_dir"`
output_filename=`basename "$3"`

command="echo '[global]\ndevice=cpu\nfloatX=float32\n' > /root/.theanorc; python program.py test /test '/output/${output_filename}'"

sudo docker run -it -v "$input_dir":/test -v "$output_dir":/output fugusuki/spacenet1 /bin/bash -c "$command"