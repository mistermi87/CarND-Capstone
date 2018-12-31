#!/usr/bin/env python
"""
This script Converts Yaml annotations to Pascal .xml Files
of the Bosch Small Traffic Lights Dataset.
Example usage:
    python bosch_to_pascal.py input_yaml out_folder
"""

import os
import sys
import yaml
from lxml import etree
import os.path
import xml.etree.cElementTree as ET


def write_xml(savedir, image, imgWidth, imgHeight,
              depth=3, pose="Unspecified"):

    boxes = image['boxes']
    impath = image['path']
    imagename = impath.split('/')[-1]
    currentfolder = savedir.split("\\")[-1]
    goalpath=os.path.normpath("D:/Bosch Dataset/Train/Rgb/Train_Jpegs/"+str(imagename))
    annotation = ET.Element("annotation")
    
    ET.SubElement(annotation, 'folder').text = 'Train_Jpegs'
    ET.SubElement(annotation, 'filename').text = str(imagename)
    ET.SubElement(annotation, 'path').text = str(goalpath) #Added in accordance witht eh output of the labelimg program
    imagename = imagename.split('.')[0]
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(imgWidth)
    ET.SubElement(size, 'height').text = str(imgHeight)
    ET.SubElement(size, 'depth').text = str(depth)
    ET.SubElement(annotation, 'segmented').text = '0'
    for box in boxes:
        boxlabel=str(box['label'])
        new_x_min=(float(box['x_min'])*600.0/710.0)-(1082-800)/2.0 # Correcting the coordinates of the bounding boxes
        new_x_max=(float(box['x_max'])*600.0/710.0)-(1082-800)/2.0
        new_y_min=(float(box['y_min'])-10)*600.0/710.0
        new_y_max=(float(box['y_max'])-10)*600.0/710.0
		#only boxes that are still within the cropped image or that show a traffic light that is actually functioning is are allow to be transformed into the new annotation file
        if not ((new_x_min<0) or  (new_x_max>800) or (new_y_min<0) or (new_y_max>600) or (boxlabel=='off')):
            obj = ET.SubElement(annotation, 'object')
			#the different labels are simplified to just "Red", "Yellow" and "Green"
            if(boxlabel=='RedLeft') or (boxlabel=='RedRight') or (boxlabel == 'RedStraight') or (boxlabel == 'RedStraightLeft') or (boxlabel == 'RedStraightRight'):
                boxlabel='Red'
            if(boxlabel=='GreenLeft') or (boxlabel=='GreenRight') or (boxlabel == 'GreenStraight') or (boxlabel == 'GreenStraightLeft') or (boxlabel == 'GreenStraightRight'):
                boxlabel='Green'
            if(boxlabel=='YellowLeft') or (boxlabel=='YellowRight') or (boxlabel == 'YellowStraight') or (boxlabel == 'YellowStraightLeft') or (boxlabel == 'YellowStraightRight'):
                boxlabel='Yellow'
				
            ET.SubElement(obj, 'name').text = boxlabel
            ET.SubElement(obj, 'pose').text = str(pose)
            ET.SubElement(obj, 'occluded').text = str(box['occluded'])
            ET.SubElement(obj, 'difficult').text = '0'

            bbox = ET.SubElement(obj, 'bndbox')
            ET.SubElement(bbox, 'xmin').text = str(round(new_x_min,3))
            ET.SubElement(bbox, 'ymin').text = str(round(new_y_min,3))
            ET.SubElement(bbox, 'xmax').text = str(round(new_x_max,3))
            ET.SubElement(bbox, 'ymax').text = str(round(new_y_max,3))

    xml_str = ET.tostring(annotation)
    root = etree.fromstring(xml_str)
    xml_str = etree.tostring(root, pretty_print=True)
    save_path = os.path.join(savedir, imagename + ".xml")
    with open(save_path, 'wb') as temp_xml:
        temp_xml.write(xml_str)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(-1)
    yaml_path = sys.argv[1]
    out_dir = sys.argv[2]
    images = yaml.load(open(yaml_path, 'rb').read())

    for image in images:
        write_xml(out_dir, image, 800, 600, depth=3, pose="Unspecified")