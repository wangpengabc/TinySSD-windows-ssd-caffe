cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
root_dir=D:\\pi_caffe

cd $root_dir

redo=1
#data_root_dir_VOC2012="E:\VOCdevkit\VOC2012"
#data_root_dir_VOC2007="E:\VOCdevkit\VOC2007"
data_root_dir="E:\\VOCdevkit"
dataset_name="VOC0712"
mapfile="$root_dir\\labelmap.prototxt"
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in test train
do
  python D:\\caffe-ssd-windows-master\\scripts\\create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $root_dir\\voc\\$subset.txt $data_root_dir\\$dataset_name\\$db\\$dataset_name"_"$subset"_"$db $dataset_name
done