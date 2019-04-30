# TinySSD-windows-ssd-caffe
## Windows Setup

* install [ caffe-ssd-windows](https://github.com/runhang/caffe-ssd-windows.git)
copy the file [create_annoset.py](https://github.com/weiliu89/caffe/tree/ssd/scripts) to caffe-ssd-windows-master/scripts
* install anaconda 
* downlaod voc datasets

## get TinySSD-windows-ssd-caffe

* download this project

## get the person dataset from the standard voc datasets

* run extra_voc_person/extra_person_2007.py  and extra_voc_person/extra_person_2012.py to extra the person data list
* run extra_voc_person to extra the name of person from xml and delete the other name object, then copy the xml to person_Annotations dirctionary 
* run voc/create_list.sh and get_image_size.bat to create the text list of person image and xml file path
* run create_data.sh to create the lmdb datasets

## train
* run TinySSD-windows-ssd-caffe/train.bat




