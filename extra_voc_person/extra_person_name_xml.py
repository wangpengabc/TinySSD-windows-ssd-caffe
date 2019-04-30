import xml.etree.cElementTree as ET
import os

# path_root = ['E:/VOCdevkit/VOC2007/Annotations',]
path_root = ['E:/VOCdevkit/VOC2012/Annotations',]

CLASSES = ["person",]
for anno_path in path_root:
    xml_list = os.listdir(anno_path)
    for axml in xml_list:
        path_xml = os.path.join(anno_path, axml)
        tree = ET.parse(path_xml)
        root = tree.getroot()

        for child in root.findall('object'):
            name = child.find('name').text
            if not name in CLASSES:
                root.remove(child)
        # print(tree)
        tree.write(os.path.join('E:/VOCdevkit/VOC2012/Annotations_person', axml))