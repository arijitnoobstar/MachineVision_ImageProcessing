
# Machine Vision & Image Processing
A series of digital image processing tools created in MATLAB from scratch. [`image_processing_report`](https://github.com/arijitnoobstar/MachineVision_ImageProcessing/blob/main/image_processing_report.pdf) contains the full details of the image processing steps and intermediate results. An example of the full image workflow is as shown:
<img src="https://github.com/arijitnoobstar/MachineVision_ImageProcessing/blob/main/image_workflow.png" width="700" />

Note that these are the example images used for this project, the text from image 1 is converted to a monochrome image

<img src="https://github.com/arijitnoobstar/MachineVision_ImageProcessing/blob/main/original_images.png" width="700" />

The following sequential steps are done using classical digital image processing techniques and coded from scratch:

 1. Conduct binary thresholding 
 2. Image segmentation using a Union-Find graph algorithm
 3. Rotation of the image using linear, bilinear & bicubic interpolation
 4. Edge detection using various techniques in the spatial domain and spatial frequency domain
 5. Finding a pixel-thin representation of the objects using mathematical morphological techniques

Image rotation is usually done by rotating the image in the reverse direction first and doing an inverse transform

<img 
src="https://github.com/arijitnoobstar/MachineVision_ImageProcessing/blob/main/image_rotation.png" width="500" />

Afterwards a series of interpolation methods can be used to interpolate the binary thresholded values into the new image

<img 
src="https://github.com/arijitnoobstar/MachineVision_ImageProcessing/blob/main/interpolation_types.png" width="500" />

For the edge detection, we used spatial frequency domain techniques like Gaussian & Butterworth filtering and in the spatial domain we used convolution-based filtering like Sobel & Prewitt operator filtering and Canny edge detection. The canny edge detection proved to be the most effective.

<img 
src="https://github.com/arijitnoobstar/MachineVision_ImageProcessing/blob/main/canny_edge.png" width="500" />

Additionally, we came up with an innovative way to use erosion via a double-element edge detection method to get the outline as well

<img 
src="https://github.com/arijitnoobstar/MachineVision_ImageProcessing/blob/main/erosion_edge_detection.png" width="500" />

Finally a series of erosion techniques with 8 operators (known as the hit-and-miss algorithm) were used to create pixel thin images.

<img 
src="https://github.com/arijitnoobstar/MachineVision_ImageProcessing/blob/main/hit_and_miss.png" width="500" />

## Collaborators
[Arijit Dasgupta](https://github.com/arijitnoobstar)
[Chong Yu Quan](https://github.com/mion666459)
