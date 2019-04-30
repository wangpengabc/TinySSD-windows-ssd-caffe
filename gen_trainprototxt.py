import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-s', '--stage', default='train', type=str,
                    help='The stage of prototxt train|test|deploy')
parser.add_argument('-d', '--lmdb', default='E:/VOCdevkit/VOC0712/lmdb/VOC0712_train_lmdb', type=str,
                    help='The lmdb datasets')
parser.add_argument('-lm', '--label_map', default='D:/pi_caffe/labelmap.prototxt', type=str,
                    help='The label_map for ssd training')
# 脚本用于生成ssd模型，这个变量可以去掉
# parser.add_argument('--generate_ssd', action='store_true',
#                     help='Default generate ssd, if this is set, generate classifier prototxt.')
# size of mobienet指的是MobileNet的宽度因子，默认为1
parser.add_argument('--size', default=1.0, type=float,
                    help='The size of mobilenet channels, support 1.0, 0.75, 0.5, 0.25.')
parser.add_argument('-c', '--class_num', default=2, type=int,
                    help='Output class number, include the \'backgroud\' class. e.g. 21 for voc.')
FLAGS = None
FLAGS, unparsed = parser.parse_known_args()
print(FLAGS)

class Generator(object):
    def __init__(self, stage, size, class_num):
        self.anchors = create_ssd_anchors()
        print(self.anchors)

        self.stage = stage
        self.size = size
        self.class_num = class_num
        self.lmdb = FLAGS.lmdb
        self.label_map = FLAGS.label_map

        # 检测是否输入数据集和label_map
        assert (self.lmdb is not None)
        assert (self.label_map is not None)

        self.input_size = 300
        self.last = "data"

    def header(self, name):
        # 第一次打开prototxt文件时，先清空
        with open('Mobie_TinySSD_train.prototxt', 'w') as f:
            f.write('name: \"%s\" \n' % name)
        f.close()

    def data_train_ssd(self):
        # 在prototxt文件中添加内容
        # tranform_param  scale:将输入从0-255变为0-2
        with open('Mobie_TinySSD_train.prototxt', 'a') as f:
            f.write("""layer {
    name: "data"
    type: "AnnotatedData"
    top: "data"
    top: "label"
    include{
        phase: TRAIN
    }
    transform_param{
        scale: 0.007843     
        mirror: true
        mean_value: 104
        mean_value: 117
        mean_value: 123
        resize_param{
          prob: 1.0
          resize_mode: WARP
          height: %d
          width: %d
          interp_mode: LINEAR
          interp_mode: AREA
          interp_mode: NEAREST
          interp_mode: CUBIC
          interp_mode: LANCZOS4
        }
        emit_constraint {
          emit_type: CENTER
        }
        distort_param {
          brightness_prob: 0.5
          brightness_delta: 32.0
          contrast_prob: 0.5
          contrast_lower: 0.5
          contrast_upper: 1.5
          hue_prob: 0.5
          hue_delta: 18.0
          saturation_prob: 0.5
          saturation_lower: 0.5
          saturation_upper: 1.5
          random_order_prob: 0.0
        }
        expand_param {
          prob: 0.5
          max_expand_ratio: 4.0
        } 
    }
    data_param {
        source: "%s"
        batch_size: 32
        backend: LMDB
    }
    annotated_data_param{
        batch_sampler {
            max_sample: 1
            max_trials: 1
        }
        batch_sampler {
            sampler {
                min_scale: 0.3
                max_scale: 1.0
                min_aspect_ratio: 0.5
                max_aspect_ratio: 2.0
            }
            sample_constraint {
                min_jaccard_overlap: 0.1
             }
          max_sample: 1
          max_trials: 50
        }
        batch_sampler {
          sampler {
            min_scale: 0.3
            max_scale: 1.0
            min_aspect_ratio: 0.5
            max_aspect_ratio: 2.0
          }
          sample_constraint {
            min_jaccard_overlap: 0.3
          }
          max_sample: 1
          max_trials: 50
        }
        batch_sampler {
          sampler {
            min_scale: 0.3
            max_scale: 1.0
            min_aspect_ratio: 0.5
            max_aspect_ratio: 2.0
          }
          sample_constraint {
            min_jaccard_overlap: 0.5
          }
          max_sample: 1
          max_trials: 50
        }
        batch_sampler {
          sampler {
            min_scale: 0.3
            max_scale: 1.0
            min_aspect_ratio: 0.5
            max_aspect_ratio: 2.0
          }
          sample_constraint {
            min_jaccard_overlap: 0.7
          }
          max_sample: 1
          max_trials: 50
        }
        batch_sampler {
          sampler {
            min_scale: 0.3
            max_scale: 1.0
            min_aspect_ratio: 0.5
            max_aspect_ratio: 2.0
          }
          sample_constraint {
            min_jaccard_overlap: 0.9
          }
          max_sample: 1
          max_trials: 50
        }
        batch_sampler {
          sampler {
            min_scale: 0.3
            max_scale: 1.0
            min_aspect_ratio: 0.5
            max_aspect_ratio: 2.0
          }
          sample_constraint {
            max_jaccard_overlap: 1.0
          }
          max_sample: 1
          max_trials: 50
        }
        label_map_file: "%s"
    }
}\n""" % (self.input_size, self.input_size, self.lmdb, self.label_map)
                    )
        f.close()

    def data_test_ssd(self):
        # 在test阶段使用的数据层
        with open('Mobie_TinySSD_train.prototxt', 'a') as f:
            f.write("""layer {
    name: "data"
    type: "AnnotatedData"
    top: "data"
    top: "label"
    include {
        phase: TEST
    }
    transform_param {
        scale: 0.007843
        mean_value: 104
        mean_value: 117
        mean_value: 123
        resize_param {
            prob: 1.0
            resize_mode: WARP
            height: %d
            width: %d
            interp_mode: LINEAR
        }
    }
    data_param {
        source: "%s"
        batch_size: 8
        backend: LMDB
    }
    annotated_data_param {
        batch_sampler {
        
        }
        label_map_file: "%s"
    }
}\n""" % (self.input_size, self.input_size, self.lmdb, self.label_map)
                    )
        f.close()

    def conv(self, name, out, kernel, stride=1, group=1, bias=False, bottom=None):
        if self.stage == "deploy": # 如果是部署阶段，需要将bn合并到bias
            bias = True

        if bottom == None:
            bottom = self.last

        padstr = ""
        if kernel > 1:
            padstr = "\n        pad: %d" % (kernel % 2)

        groupstr = ""
        if group > 1:
            groupstr = "\n        group: %d\n    engine: CAFFE" % group

        stridestr = ""
        if stride > 1:
            stridestr = "\n        stride: %d" % stride

        bias_lr_mult = ""
        bias_filler = ""
        if bias == True:
            bias_filler = """
        bias_filler{
            type: "constant"
            value: 0.0
        }"""
            bias_lr_mult = """
    param{
        lr_mult: 2.0
        decay_mult: 0.0   
    }"""

        biasstr = ""
        if bias == False:
            biasstr = "\n        bias_term: false"

        with open('Mobie_TinySSD_train.prototxt', 'a') as f:
            f.write("""layer {
    name: "%s"
    type: "Convolution"
    bottom: "%s"
    top: "%s"
    param{
        lr_mult: 1.0
        decay_mult: 1.0  
    }%s
    convolution_param {
        num_output: %d%s
        kernel_size: %d%s%s%s 
        weight_filler{
            type: "msra"
        }%s
    }
}\n""" % (name, bottom, name, bias_lr_mult, out, biasstr, kernel, stridestr, padstr, groupstr, bias_filler)
                    )
        f.close()
        self.last = name


    def bn(self, name):
        if self.stage == "deploy":
            return
        with open('Mobie_TinySSD_train.prototxt', 'a') as f:
            f.write("""layer {
    name: "%s/bn"
    type: "BatchNorm"
    bottom: "%s"
    top: "%s"
    param{
        lr_mult: 0.0
        decay_mult: 0.0
    }
}\n""" % (name, name, name,)
                    )
            f.write("""layer {
    name: "%s/scale"
    type: "Scale"
    bottom: "%s"
    top: "%s"
    param{
        lr_mult: 2.0
        decay_mult: 0.0
    }
    param{
        lr_mult: 1.0
        decay_mult: 0.0
    }       
    scale_param{
        filler{
            value: 1
        }
        bias_term: true
        bias_filler{
            value: 0
        }
    }    
}\n""" % (name, name, name)
                    )
        f.close()
        self.last = name

    def relu(self, name):
        with open('Mobie_TinySSD_train.prototxt', 'a') as f:
            f.write("""layer {
    name: "%s/relu"
    type: "ReLU"
    bottom: "%s"
    top: "%s"
}\n""" % (name, name, name)
                    )
        f.close()
        self.last = name

    def max_pooling(self, name, kernel=2, stride=2, bottom=None):
        if bottom == None:
            bottom = self.last

        with open('Mobie_TinySSD_train.prototxt', 'a') as f:
            f.write("""layer{
    name: "%s"
    type: "Pooling"
    bottom: "%s"
    top: "%s"
    pooling_param{
        pool: MAX
        kernel_size: %d
        stride: %d
    }
}\n""" %(name, bottom, name, kernel, stride)
                    )
        f.close()
        self.last = name

    def global_max_pooling(self, name, bottom=None):
        if bottom == None:
            bottom = self.last

        with open('Mobie_TinySSD_train.prototxt', 'a') as f:
            f.write("""layer{
    name: "%s"
    type: "Pooling"
    bottom: "%s"
    top: "%s"
    pooling_param{
        pool: MAX
        global_pooling: true
    }
}\n""" % (name, bottom, name)
                    )
        f.close()
        self.last = name

    def conv_bn_relu_with_factor(self, name, num, kernel, stride):
        self.num = int(num * self.size)
        self.conv(name, num, kernel, stride)
        self.bn(name)
        self.relu(name)

    def conv_bn_relu(self, name, num, kernel, stride):
        self.conv(name, num, kernel, stride)
        self.bn(name)
        self.relu(name)

    def dowm_sample_blk(self, name_conv, name_pool, num, kernel=3, stride=1):
        name1 = name_conv + "_1"
        self.conv(name1, out=num, kernel=kernel, stride=stride)
        self.bn(name1)
        self.relu(name1)

        self.conv(name_conv, out=num, kernel=kernel, stride=stride)
        self.bn(name_conv)
        self.relu(name_conv)
        self.max_pooling(name_pool)

    # def conv_dw_pw(self):
    def permute(self, name):
        with open('Mobie_TinySSD_train.prototxt', 'a') as f:
            f.write("""layer {
    name: "%s_perm"
    type: "Permute"
    bottom: "%s"
    top: "%s_perm"
    permute_param {
        order: 0
        order: 2
        order: 3
        order: 1
    }
}\n""" % (name, name, name)
                    )
        f.close()
        self.last = name + "_perm"

    def flatten(self, name):
        with open('Mobie_TinySSD_train.prototxt', 'a') as f:
            f.write("""layer {
    name: "%s_flat"
    type: "Flatten"
    bottom: "%s_perm"
    top: "%s_flat"
    flatten_param {
        axis: 1
    }
}\n""" % (name, name, name)
                    )
        f.close()
        self.last = name + "_flat"

    def mbox_priors(self, name, min_size, max_size, aspect_ratio):
        min_box = min_size * self.input_size
        max_box = max_size * self.input_size
        aspect_ratio_str = ""
        for ar in aspect_ratio:
            aspect_ratio_str += "\n        aspect_ratio: %.1f" % ar

        with open('Mobie_TinySSD_train.prototxt', 'a') as f:
            f.write("""layer {
    name: "%s_mbox_priorbox"
    type: "PriorBox"
    bottom: "%s"
    bottom: "data"
    top: "%s_mbox_priorbox"
    prior_box_param{
        min_size: %.1f
        max_size: %.1f%s
        flip: true
        clip: false
        variance: 0.1
        variance: 0.1
        variance: 0.2
        variance: 0.2
        offset: 0.5
    }
}\n""" % (name, name, name, float(min_box), float(max_box), aspect_ratio_str)
                    )
        f.close()

    def mbox_loc(self, bottom, num):
        name = bottom + "_mbox_loc"
        self.conv(name, out=num, kernel=1, bias=True, bottom=bottom)
        self.permute(name)
        self.flatten(name)

    def mbox_conf(self, bottom, num):
        name = bottom + "_mbox_conf"
        self.conv(name, out=num, kernel=1, bias=True, bottom=bottom)
        self.permute(name)
        self.flatten(name)

    def mbox(self, bottom, num_box):
        self.mbox_loc(bottom, num_box * 4)
        self.mbox_conf(bottom, num_box * self.class_num)
        min_size, max_size = self.anchors[0]
        self.mbox_priors(bottom, min_size, max_size, aspect_ratio=[2.0, 3.0]) # 设置ratio为1 2 0.5，每个特征点生成4个锚框

        self.anchors.pop(0) # 移除索引为0的list

    def concat_boxes(self, convs):
        for layer in ["loc", "conf"]:
            bottom = ""
            for cnv in convs:
                bottom += "\n    bottom: \"%s_mbox_%s_flat\"" % (cnv, layer)
            with open('Mobie_TinySSD_train.prototxt', 'a') as f:
                f.write("""layer {
    name: "mbox_%s"
    type: "Concat"%s
    top: "mbox_%s" 
    concat_param{
        axis: 1
    }
}\n""" % (layer, bottom, layer)
                        )
            f.close()

        bottom = ""
        for cnv in convs:
            bottom += "\n    bottom: \"%s_mbox_priorbox\"" % cnv

        with open('Mobie_TinySSD_train.prototxt', 'a') as f:
            f.write("""layer {
    name: "mbox_priorbox"
    type: "Concat"%s
    top: "mbox_priorbox" 
    concat_param{
        axis: 2
    }
}\n""" % bottom
                    )
        f.close()

    def ssd_loss(self):
        with open('Mobie_TinySSD_train.prototxt', 'a') as f:
            f.write("""layer{
    name: "mbox_loss"
    type: "MultiBoxLoss"
    bottom: "mbox_loc"
    bottom: "mbox_conf"
    bottom: "mbox_priorbox"
    bottom: "label"
    top: "mbox_loss"
    include {
        phase: TRAIN
    }
    propagate_down: true
    propagate_down: true
    propagate_down: false
    propagate_down: false
    loss_param {
        normalization: VALID
    }
    multibox_loss_param {
        loc_loss_type: SMOOTH_L1
        conf_loss_type: SOFTMAX
        loc_weight: 1.0
        num_classes: %d
        share_location: true
        match_type: PER_PREDICTION
        overlap_threshold: 0.5
        use_prior_for_matching: true
        background_label_id: 0
        use_difficult_gt: true
        neg_pos_ratio: 3.0
        neg_overlap: 0.5
        code_type: CENTER_SIZE
        ignore_cross_boundary_bbox: false
        mining_type: MAX_NEGATIVE
    }
}\n"""% self.class_num
                    )
        f.close()

    def ssd_test(self):
        with open('Mobie_TinySSD_train.prototxt', 'a') as f:
            f.write("""layer {
    name: "mbox_conf_reshape"
    type: "Reshape"
    bottom: "mbox_conf"
    top: "mbox_conf_reshape"
    reshape_param {
        shape {
        dim: 0
        dim: -1
        dim: %d
        }
    }
}
layer {
    name: "mbox_conf_softmax"
    type: "Softmax"
    bottom: "mbox_conf_reshape"
    top: "mbox_conf_softmax"
    softmax_param {
    axis: 2
    }
}
layer {
    name: "mbox_conf_flatten"
    type: "Flatten"
    bottom: "mbox_conf_softmax"
    top: "mbox_conf_flatten"
    flatten_param {
        axis: 1
    }
}
layer {
    name: "detection_out"
    type: "DetectionOutput"
    bottom: "mbox_loc"
    bottom: "mbox_conf_flatten"
    bottom: "mbox_priorbox"
    top: "detection_out"
    include {
        phase: TEST
    }
    detection_output_param {
    num_classes: %d
    share_location: true
    background_label_id: 0
    nms_param {
        nms_threshold: 0.45
        top_k: 400
    }
    code_type: CENTER_SIZE
    keep_top_k: 200
    confidence_threshold: 0.01
    }
}
layer {
    name: "detection_eval"
    type: "DetectionEvaluate"
    bottom: "detection_out"
    bottom: "label"
    top: "detection_eval"
    include {
        phase: TEST
    }
    detection_evaluate_param {
        num_classes: %d
        background_label_id: 0
        overlap_threshold: 0.5
        evaluate_difficult_gt: false
    }
}\n""" % (self.class_num, self.class_num, self.class_num)
                    )
        f.close()

    def generate(self):
        self.header('MobileNet_SSD')
        if self.stage == "train":
            self.data_train_ssd()
        elif self.stage == "test":
            self.data_test_ssd()
        # else:
        #     self.data_deploy_ssd()

        self.dowm_sample_blk("conv1", "pool1", 16)
        self.dowm_sample_blk("conv2", "pool2", 32)
        self.dowm_sample_blk("conv3", "pool3", 64)
        self.dowm_sample_blk("conv4", "pool4", 128)
        self.dowm_sample_blk("conv5", "pool5", 128)
        self.dowm_sample_blk("conv6", "pool6", 128)
        self.global_max_pooling("pool")
        self.mbox("pool3", 6) # 每个特征点生成6个锚框
        self.mbox("pool4", 6)
        self.mbox("pool5", 6)
        self.mbox("pool6", 6)
        self.mbox("pool", 6)
        self.concat_boxes(["pool3", "pool4", "pool5", "pool6", "pool"])
        if self.stage == "train":
            self.ssd_loss()
        elif self.stage == "test":
            self.ssd_test()

# 返回不同scales的框，范围从0-1，生成锚框过程中定义锚框大小
# 返回为六行一行两列的数组
def create_ssd_anchors(num_layers=6, min_scale=0.2, max_scale=0.9):
    scales = [min_scale + (max_scale - min_scale) * i / num_layers
              for i in range(num_layers)] + [0.95]

    return list(zip(scales[:-1], scales[1:]))


if __name__ == '__main__':
    gen = Generator(FLAGS.stage, FLAGS.size, FLAGS.class_num)
    gen.generate()
