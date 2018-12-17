### Minimizing the Bosch Dataset
The aim of this workflow is the creation of a compact Bosch Dataset for traffic light detection that can be used in conjunction with images from the simulator and images from the video stream of the Udacity self driving car ("ROSbag"). For this the Bosch dataset has to be altered in several ways:
- The image reslution has to be reduced from 1280x720 to 800x600
- The image type has to be changed to jpeg from png (and simultaniously reduced in filesize for better handling)
- The annotations of the images have to be reduced from specific traffic light types to the basic colors in accordance with the other datasets (e.g. "Redleft", "Redstraight" to "Red")
- The bounding boxes in the annotations need to be adjusted for the crop and zoom of the images
#### Transforming the images
The original Bosch Traffic Light Dataset contained 12bit HDR images created with red-clear-clear-blue filter and 8bit RGB conversions. We used the converted images to stay have comparable data to our other datasets. However, the conversion had some drawbacks:
- yellowish color tint
- low contrast range, dark
- purple color fringe at the top of the image
- purple color fringe in bright reflections or in bright sky

In order to create images similar to our other images and mitigate some of the drawbacks we perfomed a batch operation in Adobe Photoshop (but could this could be done in GIMP as well):

 1. Cut the top 10 px from the original image, to get rid of the purple fringe at the top - reducing the vertical resolution to 710 px
 2. Change the overall resolution, such that the vertical resolution becomes 600 px, - creating an image of 1082x600 px
 3. Cut the image symmetrical to the left and right, so that the final resolution of the image becomes 600x800 px
 4. Use Auto Color / Auto Tone / Auto Contrast on each image. This yields varying results for each image, which for our case could help generalize the overall data. For consistent image correction following color-curves have to be altered: Bright reds --> brighter, overall blues (with emphasis on medium bright)--> brighter. Optionally, the dark areas could be brightened slightly using the RGB curve. This prevents loss of detail when the low pass filtering of the jpeg reduction is applied to the image
 5. Save image in jpeg format. In Photoshop a strength of 5(/10) was used. This yields to an overall imagesize of 45-70kb

#### Transforming the annotation

First of all, all instances of `[img...].png` are replaced to `[img...].jpg`  in the original `train.yaml`. Then the `bosch_to_pascal.py` provided by the the Bosch Repo https://github.com/bosch-ros-pkg/bstld is edited :
 - a line is added that creates the < path > <\ path> line in accordance with the output of labelImg 
```python
ET.SubElement(annotation, 'path').text = str(goalpath)

```
- The new dimension of the bounding boxes are calculated. The transformation that was before applied to the images is now applied mathematically to the boxes. Additionally the label of the box is read
```python
boxlabel=str(box['label'])
new_x_min=(float(box['x_min'])*600.0/710.0)-(1082-800)/2.0
new_x_max=(float(box['x_max'])*600.0/710.0)-(1082-800)/2.0
new_y_min=(float(box['y_min'])-10)*600.0/710.0
new_y_max=(float(box['y_max'])-10)*600.0/710.0
```
- With this info, only boxes that are still within the cropped image or that show a traffic light that is actually functioning is are allow to be transformed into the new annotation file
```python
if not ((new_x_min<0) or  (new_x_max>800) or (new_y_min<0) or (new_y_max>600) or (boxlabel=='off')):
	obj = ET.SubElement(annotation, 'object')
...
```
- the different labels are simplified to just "Red", "Yellow" and "Green"
```python
if(boxlabel=='RedLeft') or (boxlabel=='RedRight') or (boxlabel == 'RedStraight') or (boxlabel == 'RedStraightLeft') or (boxlabel == 'RedStraightRight'):
	boxlabel='Red'
if(boxlabel=='GreenLeft') or (boxlabel=='GreenRight') or (boxlabel == 'GreenStraight') or (boxlabel == 'GreenStraightLeft') or (boxlabel == 'GreenStraightRight'):
	boxlabel='Green'
if(boxlabel=='YellowLeft') or (boxlabel=='YellowRight') or (boxlabel == 'YellowStraight') or (boxlabel == 'YellowStraightLeft') or (boxlabel == 'YellowStraightRight'):
	boxlabel='Yellow'
...
```
- In the final line the resolution of the input image is adjusted
```python
for image in images:
	 write_xml(out_dir, image, 800, 600, depth=3, pose="Unspecified")
```

Finally the file is executed in its folder as advised:
`python boschmini_to_pascal.py input_yaml out_folder`