% ME5405: Machine Vision Project 
% AY 20/21 Semester 1
% Group 26
% Authors: Arijit Dasgupta, Chong Yu Quan
% This code handles the image processing operations for Image 1. To anybody
% reading the code, you can change the parameters under the CONFIG section
% below. Binary Threshold is set in the corresponding step

clear all;

%--------------------------------------------------------------------------

%%%---CONFIG---%%%
segmentation_method = 'classical'; % iterative, classical
interpolation_method = 'bicubic'; % nearest neighbour, bilinear, bicubic 
segmented_image_length = 21; % length of segmented image (as a square image)
filtering_domain = 'spatial'; % frequency, spatial, morphology
frequency_filtering_method = "gaussian"; % butterworth, gaussian
spatial_filtering_method = "canny"; % prewitt, sobel, canny
%%%---END OF CONFIG---%%%

%--------------------------------------------------------------------------


% read the input txt image
im = char(textread("charact1.txt", '%s','delimiter',''));

rows = size(im, 1); % 64
cols = size(im, 2); % 64
colours = size(im, 3); % 1 as the image is converted to grayscale
levels = 256; % number of quantized levels

%%%%%%%%%%% STEP 0 %%%%%%%%%%%%%%%%

% Before proceeding the image processing for Image 1, we must convert the
% 32 level (0-9 & A-Z) into a 0 - 255 scale intensity image in grayscale.
% This can be done by mapping each one of the curernt mappings from 1 to
% 32. After that we multiply it by (256/32) = 8, and then subtract by one.

% First we create a mapping from the 0-9 & A-V to 0-255 using MATLAB's
% containers.Map object
% keyset contains the 0-9 and A-Z and this is hardcoded for charact1.txt
keySet = {'0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F'...
    ,'G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V'};
% we manually code out the value set as the 0-255 mapping
valueSet = zeros(1, size(keySet,2), 'uint8');
for i = 1:size(keySet,2)
    valueSet(i) = (i - 1) * uint8(levels/size(keySet,2));
end

% create the mapping
map = containers.Map(keySet, valueSet);

% We create a dummy array to store the pixel values of the converted image
im_temp = zeros(rows, cols, 'uint8');
% Then we loop through the current image and conduct the mapping
for i = 1:rows
    for j = 1:cols
        im_temp(i,j) = map(im(i,j));
    end
end
% Equate the im_temp back to im
im = im_temp;

%%%%%%%%%%% STEP 1 %%%%%%%%%%%%%%%%
% Display the image, using the display_image function we wrote
figure(1)
display_image(im, "Image before processing");

%%%%%%%%%%% STEP 2 %%%%%%%%%%%%%%%%
% To conduct binary thresholding, we must first come up with a threshold
% We can start of by visualising the histogram of the image

H = get_histogram(im, levels);

figure(2);
display_histogram(H, "Histogram of Image before Binary Thresholding");


threshold = 10;
% Binary Threshold the image
for i = 1:rows
    for j = 1:cols
        if im(i,j) <= threshold
            im(i,j) = 0;
        else
            im(i,j) = 255;
        end
    end
end
% Show the thresholded image
figure(3)
display_image(im, "Binary Image after Thresholding");

%%%%%%%%%%% STEP 3 %%%%%%%%%%%%%%%%
% The next step is to do image segmentation where we segment the different
% characters out of the image and separate them. Our approach to
% segementing the different characters is by applying connected components
% algorithms to identify and label the different characters. This requires
% the accumption that no two characters are connected. We can already
% verify this assumption visually. The algorithms implemented will assume a
% 4 connectivity level of connection.

if segmentation_method == "iterative"
    %---- Method 3A: Iterative Algorithm -----%

    % Create the image array for the labels, background will be labelled as 0
    % and the object pixels will get labels from 1 to N where N is the number
    % of detected objects
    labelled_im = zeros(rows, cols, 'uint64');

    % Initialise the array with unique labels for each pixel labelled 255
    label = 1;
    for i = 1:rows
        for j = 1:cols
            if im(i,j) == 255
                labelled_im(i,j) = label;
                label = label + 1;
            end
        end
    end

    % Run the algorithm until the labelled_im array stops changing
    % num_passes variable to check number of iterations (of double-passes)
    % needed for convergence
    num_passes = 0;
    % temp variable used as a checked to see when the algorithm should stop
    % running
    labelled_im_temp = zeros(rows, cols, 'uint64');
    % Now implement the first pass of the algorithm in top-down and left-right
    % direction while checking only the top and left neighouring
    % pixels
    while ~isequal(labelled_im,labelled_im_temp)
        labelled_im_temp = labelled_im;
        num_passes = num_passes + 1;
        for i = 1:rows
            for j = 1:cols
                if im(i,j) == 255
                    % create an array to store values of neighbouring object pixels
                    values = [Inf, Inf];
                    % two checks are done in the top and left neighbouring pixels, first if the
                    % coordinate is valid in the image and the second is to check
                    % if the pixel is an object pixel
                    if i - 1 > 0
                        if im(i-1,j) == 255
                            values(1) = labelled_im(i-1,j);
                        end
                    end
                    if j - 1 > 0
                        if im(i,j-1) == 255
                            values(2) = labelled_im(i,j-1);
                        end
                    end
                    % Replace the label with the minimum label value of neighbours
                    if min(values) ~= Inf    % in case the pixel is completely isolated, it will keep its original label
                        labelled_im(i,j) = min(values);
                    end
                end
            end
        end

        % Now do the second pass in the exact opposite direction (bottom-up and
        % right-left) while checking only the right and bottom neighouring
        % pixels
        for i = rows:-1:1
            for j = cols:-1:1
                if im(i,j) == 255
                    % create a cell array to store values of neighbouring object pixels
                    values = [Inf, Inf];
                    % two checks are done in the bottom and right neighbouring pixels, first if the
                    % coordinate is valid in the image and the second is to check
                    % if the pixel is an object pixel
                    if i + 1 <= rows
                        if im(i+1,j) == 255
                            values(1) = labelled_im(i+1,j);
                        end
                    end
                    if j + 1 <= cols
                        if im(i,j+1) == 255
                            values(2) = labelled_im(i,j+1);
                        end 
                    end
                    % Replace the label with the minimum label value of neighbours
                    if min(values) ~= Inf    % in case the pixel is completely isolated, it will keep its original label
                        labelled_im(i,j) = min(values);
                    end
                end
            end
        end
    end

    % Great! It took 5 passes (top-down and bottom-up is one pass) for the
    % algorithm to converge
    
elseif segmentation_method == "classical"

    %---- Method 3B: Classical Algorithm -----%

    % Create the image array for the labels
    labelled_im = zeros(rows, cols, 'uint64');

    % create equivalences array to record all equivalent labels
    equivalences = [];
    % initialise the label to 0
    label = 0;
    % top-down & left-right pass
    for i = 1:rows
        for j = 1:cols
            if im(i,j) == 255
                % create an array to store values of neighbouring object pixels
                values = [Inf, Inf];
                % two checks are done in the top and left neighbouring pixels, first if the
                % coordinate is valid in the image and the second is to check
                % if the pixel is an object pixel
                if i - 1 > 0
                    if im(i-1,j) == 255
                        values(1) = labelled_im(i-1,j);
                    end
                end
                if j - 1 > 0
                    if im(i,j-1) == 255
                        values(2) = labelled_im(i,j-1);
                    end
                end
                % check if both top and left cells are Inf (both cells are
                % either invalid or a background point)
                if min(values) == Inf
                    % increment the label and label current cell with it
                    label = label + 1;
                    labelled_im(i,j) = label;
                % otherwise, replace the label with the minimum label value of neighbours
                else
                    labelled_im(i,j) = min(values);
                    % check if both labels are the different (and not Inf), if so, add them to
                    % equivalence array
                    if values(1) ~= values(2) && values(1) ~= Inf && values(2) ~= Inf
                        % Check to ensure this equivalence is not already in
                        % the array. An empty equivalence array passes the condition too
                        if isequal(equivalences, []) || ~ismember([min(values) max(values)], equivalences, 'rows')
                            % arrange equivalence such that the minimum number is
                            % shown first
                            equivalences = [equivalences ; [min(values) max(values)]];
                        end
                    end
                end
            end
        end
    end


    %---- GRAPH SEARCH ALGORITHM -----%
    % Now we shall implement a graph search algoritnm to do a union-find over
    % the disjointed set of labels. Each label present in labelled_im (excluding
    % 0) is treated as its own disjointed set. The equivalences are used as
    % instructions for joining the disjointed sets.

    % Create array of the unique labels present in the equivalences set
    unique_labels = unique(equivalences);

    % create a array to store all nodes of the graph
    graph_nodes = [];

    % Loop through each unique label and make the graph structure by 
    % initialising all nodes with the parent attribute to itself with a size of 1
    for i = 1:size(unique_labels, 1)
        % Initialise LabelNode, a class we created to save the node attributes
        node = LabelNode;
        % set the size to 1
        node.Size = 1;
        % save the label it is representing
        node.Label = unique_labels(i);
        % self reference as the parent node. This is possible with the use of
        % the handle class in MATLAB
        node.Parent = node;
        % add the node to the graph_nodes array
        graph_nodes = [graph_nodes ; node];
    end

    % using the equivalences as instructions, we can pass them down one by one
    % to join two disjointed sets into the same graph
    for i = 1:size(equivalences, 1)
        node_A = return_node_by_label(equivalences(i,1), graph_nodes);
        node_B = return_node_by_label(equivalences(i,2), graph_nodes);
        Union(node_A, node_B);
    end

    %xxxx END OF GRAPH SEARCH ALGORITHM xxxx%

    % Now do the second pass to use the equivalences and relabel the entire
    % image
    for i = rows:-1:1
        for j = cols:-1:1
            if im(i,j) == 255  
                % find the node associated with the label in this pixel
                node = return_node_by_label(labelled_im(i,j), graph_nodes);
                % in case no node is found, the object pixel may be
                % completely isolated and not involved in any equivalences
                if node == "no node"
                    continue
                end
                % find the parent node, which would represent the final label
                root_node = Find(node);
                % reassign label as such, this would work even if root_node and
                % node are the same
                labelled_im(i,j) = root_node.Label;
            end
        end
    end
end


%---- POST ALGORITHM SEGMENTATION -----%
% By this stage the connected components algorithms have been run and it is
% time to segment the different characters into separate images


number_of_components = size(unique(labelled_im),1);
unique_labels = unique(labelled_im);
% create cell array to save the images
segmented_im = {};

% loop through each label
for lbl = 1:number_of_components
    label = unique_labels(lbl);
    % if label is 0, ignore the label as we don't care about the background
    if label == 0
        continue
    end
    % extract i,j coordinates for the segment
    coordinates = [];
    for i = 1:rows
        for j = 1:cols
            if labelled_im(i,j) == label
                coordinates = [coordinates ; [i,j]];
            end
        end
    end
    % find the max and min coordinates for x and y of the segment
    min_x = min(coordinates(:,1));
    max_x = max(coordinates(:,1));
    min_y = min(coordinates(:,2));
    max_y = max(coordinates(:,2));
    % For the new image, we shall use a size of 41 x 41 pixels standardised
    % for all characters, hence the midpoint of the new image (21,21) will
    % correspond to the halfway point of the min and max coordinates of the
    % segment
    
    % create image array and deltas needed to map the coordinates from the
    % full image to the new image
    new_image = zeros(segmented_image_length,segmented_image_length,'uint8');
    delta_x = round((max_x + min_x)/2) - round(segmented_image_length/2);
    delta_y = round((max_y + min_y)/2) - round(segmented_image_length/2);
    for i =  1 : size(coordinates, 1)
        new_image(coordinates(i,1) - delta_x, coordinates(i,2) - delta_y) = 255;
    end
    % add the image into the segmented_im cell array
    segmented_im{end + 1} = new_image;
end

%---- VISUALISE THE SEGMENTED CHARACTERS -----%
% Now let us look through each one of the segmented images and show them in
% one figure with white pixel-thin lines to show segmentation
combined_segmented_img = combine_segmented_images(segmented_im);

figure(4);
display_image(combined_segmented_img, "Segmented Characters using the " + segmentation_method + " method");

%%%%%%%%%%% STEP 4 %%%%%%%%%%%%%%%%
% The next two steps involve rotating the segmented characters by a certain
% angle about their respective centriods. For the first step, we have to rotate the image by 90 degrees
% clockwise. For both steps, the first step is to identify the centroids of
% the image, after which, the image is rotated about the centroid using the
% rotation matrix and subsequently followed by brightness interpolation

% set clockwise angle for rotation
%----------------------------%
angle = 90;                  %
%----------------------------%

% Get an array of centroids (i,j) for all segmented images
% value used in binary thresholding is used as a criteria for considering
% which pixels count in centroid calculation. As the image is already
% binary, this threshold won't matter
centroids = determine_centroids(segmented_im, threshold);

% The next step is to find the i and j positions of the object pixels for
% the inverse transform of the rotation on the new image
% These positions are likely to be non-integer values
% backward_mapping_coordinates contains the coordinates for the
% "rotated image" if it were to be rotated in the opposite manner for
% backward mapping
backward_mapping_coordinates = determine_backward_mapping_coordinates(segmented_im, centroids, angle);

% In the next step, we will apply interpolation and create the new rotated
% images
segmented_im_90deg = brightness_interpolation(segmented_im, backward_mapping_coordinates, interpolation_method);

%---- VISUALISE THE ROTATED CHARACTERS -----%
combined_segmented_img = combine_segmented_images(segmented_im_90deg);

figure(5);
display_image(combined_segmented_img, "Segmented Characters rotated clockwise by " + angle + ...
    " degrees about their centroids using " + interpolation_method + " brightness interpolation");

%%%%%%%%%%% STEP 5 %%%%%%%%%%%%%%%%

% Step 5 involves the exact same steps as step 4, except that we are
% rotating about a new set of centroids counterclockwise by 35 degrees
% using the new set of rotated images from step 4

angle = -35;                  
% The value used in binary thresholding is used as a criteria for considering
% which pixels count in centroid calculation. As the image has been
% interpolated, it is no longer binary and this criteria is needed.
centroids = determine_centroids(segmented_im_90deg, threshold);

backward_mapping_coordinates = determine_backward_mapping_coordinates(segmented_im_90deg, centroids, angle);

segmented_im_35deg = brightness_interpolation(segmented_im_90deg, backward_mapping_coordinates, interpolation_method);

combined_segmented_img = combine_segmented_images(segmented_im_35deg);

figure(6);
display_image(combined_segmented_img, "90-degree rotated Segmented Characters rotated clockwise by " + angle + ...
    " degrees about their centroids using " + interpolation_method + " brightness interpolation");

%%%%%%%%%%% STEP 6 %%%%%%%%%%%%%%%%
% The focus of this step is edge detection to determine the outline(s) of
% the characters. There are multiple ways of doing this. The non-rotated
% images will be used here

if filtering_domain == "frequency"
    %---- Method 6A: High-pass filtering in the spatial frequency domain -----%
    % For this method, we will use discrete fast fourier transform to convert
    % the segmented images into the frequency domain. After that we will apply
    % a high pass filter to suppress all details that don't correspont to
    % edges. Then we will do an inverse fourier transform to recover the image

    % three types of filters can be used in the frequency domain: ideal,
    % Butterworth and Gaussian

    % create cell array to store images of the outlines including the FFT
    % shifted versions
    segmented_im_outline = {};
    F_segmented_im_outline = {};
    % set D0 value for gaussian and butterworth filtering
    D0 = 2.4;
    %n value for butterworth filtering
    n = 1;

    for count = 1:size(segmented_im,2)
        image = im2double(cell2mat(segmented_im(count)));
        % use the trick taught in class to centre the FFT
        for i=1:size(image,1)
            for j=1:size(image,2)
               image_centered(i,j)=((-1)^(i+j)).*image(i,j);
            end
        end

        % apply discrete fft and shift the zero-frequency to the centre
        F_image = fft2(image_centered);

        % apply filter to the freqencey domain
        H = zeros(size(image_centered,1), size(image_centered,2), 'double');
        for u = 1:size(image_centered,1)
            for v = 1:size(image_centered,2)
                D = ((u-size(image_centered,1)/2)^2+(v-size(image_centered,2)/2)^2)^(1/2);
                if frequency_filtering_method == "butterworth"
                    % butterworth filtering
                    H(u,v)= 1 - (1/(1+(D/D0)^(2*n)));
                elseif frequency_filtering_method == "gaussian"
                    % gaussian filtering
                    H(u,v)= 1- (exp(-(D^2)/(2*(D0^2))));
                elseif frequency_filtering_method == "ideal"
                    % ideal filtering (using D0 as cutoff)
                    if D <= D0
                        H(u,v) = 1;
                    else
                        H(u,v) = 0;
                    end
                end
            end
        end      
        % add the filter to the image in the spatial frequency domain
        F_filtered_image = F_image.*H;

        % inverse discrete fft to get the image back (not yet de-centered)
        filtered_image=real(ifft2(F_filtered_image));

        % de-center the image using the trick taught in class
        for i=1:size(image,1)
            for j=1:size(image,2)
                filtered_image(i,j)=((-1)^(i+j)).*filtered_image(i,j);
            end
        end

        % some touching up
        filtered_image = uint8(filtered_image);
        filtered_image(filtered_image>=1) = 255;
        filtered_image(filtered_image<0.0001) = 0;

        a = log(abs(F_image)+1);
        F_segmented_im_outline{end+1} = a;
        segmented_im_outline{end+1} = filtered_image;

    end

elseif filtering_domain == "spatial"
    %---- Method 6B: High-pass filtering in the spatial domain -----%
    % Now we shall try high-pass filters over the original image, i.e. the
    % spatial domain
    
    segmented_im_outline = {};

    for count = 1:size(segmented_im,2)
        image = im2double(cell2mat(segmented_im(count)));
        % we chose to use the built-in edge filtering operation in MATLAB
        % for the spatial filtering
        filtered_image = edge(image, spatial_filtering_method);
        segmented_im_outline{end+1} = filtered_image;
    end
    % for use in the title of the visualisation
    frequency_filtering_method = spatial_filtering_method;
    
elseif filtering_domain == "morphology"
    %---- Method 6C: Erosion with a NxN structured element -----%
    % Now we shall use mathematical morphology to erode the boundary away
    % with a structured element and subtract the output from the original 
    % image to get an outline. This is known as boundary extraction
    
    segmented_im_outline = {};

    for count = 1:size(segmented_im,2)
        image = im2double(cell2mat(segmented_im(count))); 
        % create a structured element object (MATLAB built-in)origin: top
        % left
        SE_1 = strel([1 1;1 1]);
        % create a structured element object (MATLAB built-in)origin:
        % bottom right
        SE_2 = strel([1 1 0; 1 1 0; 0 0 0]);
        % erode away the image with the structured element
        image_eroded_1 = imerode(image, SE_1);
        image_eroded_2 = imerode(image, SE_2);
        % invert the process by subtracting it from the original image to
        % get the outline only using both SEs (detailed in the report)
        image_outline = image - image_eroded_1 + image - image_eroded_2;
        segmented_im_outline{end+1} = image_outline;
    end
    % for use in the title of the visualisation
    frequency_filtering_method = "morphology (erosion)";
    filtering_domain = "spatial";
end

%----VISUALISE THE OUTLINED IMAGES-----%
combined_segmented_img = combine_segmented_images(segmented_im_outline);
figure(7);
display_image(combined_segmented_img, "Segmented Characters after high-pass filtering using the "...
    + frequency_filtering_method + " filtering method in the " + filtering_domain + " domain");

%%%%%%%%%%% STEP 7 %%%%%%%%%%%%%%%%
% In this stage, the goal is to achieve thinning of the images by using a
% thinning operation via mathematical morphology to achieve a pixel-thin
% line representation of every character. This thinning is done by using
% the hit and miss morphological operation. A total of 2 different
% structred hit-and-miss elements are used. A reference for this can be
% found at https://homepages.inf.ed.ac.uk/rbf/HIPR2/thin.htm

segmented_im_thin = {};

% array for the structured element. 1 refers to pixels that are used for
% morphology whole 0 is ignored
SE1_nhood_hit = [0 0 0 ; 0 1 0 ; 1 1 1];
SE1_nhood_miss = [1 1 1 ; 0 0 0 ; 0 0 0];
SE2_nhood_hit = [0 0 0 ; 1 1 0 ; 0 1 0];
SE2_nhood_miss = [0 1 1 ; 0 0 1 ; 0 0 0];

% create the hit and miss structured elements. hit refers to pixels that
% should match the object while miss refers to elements that should match
% the background. Other unspecified pixels are ignored. The structured
% elements in all 4 90 degree rotations are created
SE1hit_0 = strel('arbitrary', imrotate(SE1_nhood_hit, 0));
SE1miss_0 = strel('arbitrary', imrotate(SE1_nhood_miss, 0));
SE1hit_90 = strel('arbitrary', imrotate(SE1_nhood_hit, -90));
SE1miss_90 = strel('arbitrary', imrotate(SE1_nhood_miss, -90));
SE1hit_180 = strel('arbitrary', imrotate(SE1_nhood_hit, -180));
SE1miss_180 = strel('arbitrary', imrotate(SE1_nhood_miss, -180));
SE1hit_270 = strel('arbitrary', imrotate(SE1_nhood_hit, -270));
SE1miss_270 = strel('arbitrary', imrotate(SE1_nhood_miss, -270));

SE2hit_0 = strel('arbitrary', imrotate(SE2_nhood_hit, 0));
SE2miss_0 = strel('arbitrary', imrotate(SE2_nhood_miss, 0));
SE2hit_90 = strel('arbitrary', imrotate(SE2_nhood_hit, -90));
SE2miss_90 = strel('arbitrary', imrotate(SE2_nhood_miss, -90));
SE2hit_180 = strel('arbitrary', imrotate(SE2_nhood_hit, -180));
SE2miss_180 = strel('arbitrary', imrotate(SE2_nhood_miss, -180));
SE2hit_270 = strel('arbitrary', imrotate(SE2_nhood_hit, -270));
SE2miss_270 = strel('arbitrary', imrotate(SE2_nhood_miss, -270));

for count = 1:size(segmented_im,2)
    image = im2double(cell2mat(segmented_im(count)));
    % run the algorithm until image and image_temp are the same, i.e. the
    % algorithm has converged
    image_temp = zeros(size(image,1),size(image,2), 'double');
    while ~isequal(image_temp,image)
        image_temp = image;
        % carry out the hit-and-miss operations for all 90 degree angles
        % for for streuctured elements
        image = image - (imerode(image,SE1hit_0) & imerode(~image,SE1miss_0));
        image = image - (imerode(image,SE2hit_0) & imerode(~image,SE2miss_0));

        image = image - (imerode(image,SE1hit_90) & imerode(~image,SE1miss_90));
        image = image - (imerode(image,SE2hit_90) & imerode(~image,SE2miss_90));
        
        image = image - (imerode(image,SE1hit_180) & imerode(~image,SE1miss_180));
        image = image - (imerode(image,SE2hit_180) & imerode(~image,SE2miss_180));
        
        image = image - (imerode(image,SE1hit_270) & imerode(~image,SE1miss_270)); 
        image = image - (imerode(image,SE2hit_270) & imerode(~image,SE2miss_270));
    end
    % algorithm has converged, not add the image to the cell array
    segmented_im_thin{end+1} = image;
end

%----VISUALISE THE PIXEL-THIN IMAGES-----%
combined_segmented_img = combine_segmented_images(segmented_im_thin);
figure(8);
display_image(combined_segmented_img, "Segmented Characters after single-pixel thinning using "...
    + "mathematical morphology");

%%%%%%%%%%% STEP 8 %%%%%%%%%%%%%%%%
% In this final step, the characters simply have to be put in order
segmented_im_ordered = [];
% Character A
segmented_im_ordered = [segmented_im_ordered cell2mat(segmented_im(5))];
% Character 1
segmented_im_ordered = [segmented_im_ordered cell2mat(segmented_im(1))];
% Character B
segmented_im_ordered = [segmented_im_ordered cell2mat(segmented_im(6))];
% Character 2
segmented_im_ordered = [segmented_im_ordered cell2mat(segmented_im(2))];
% Character C
segmented_im_ordered = [segmented_im_ordered cell2mat(segmented_im(4))];
% Character 3
segmented_im_ordered = [segmented_im_ordered cell2mat(segmented_im(3))];

figure(9);
display_image(segmented_im_ordered, "Segmented Characters in specified order");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% FUNCTIONS %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% function to display image
function display_image(im, title_name)
    imshow(im);
    % display the image in a figure
    axis on, axis image, colorbar;
    % add title
    title(title_name);
end

% Function to get histogram array from image
function H = get_histogram(im, levels)
    % Create empty histogram array with length levels (256)
    H = zeros(1, levels, 'uint64');
    % loop through all pixels in the image
    for i=1:size(im,1)
        for j=1:size(im,2)
            % increment the repective histogram identifying value, a +1 is
            % needed as pixel values range from 0 to 255 while in matlab, the
            % indexing starts from 1
            H(im(i,j) + 1) = H(im(i,j) + 1) + 1;
        end
    end
end

function display_histogram(H, title_name)
    % Visualise the Histogram
    bar(H);
    ax = gca;
    ax.Visible = 'On';
    xlabel("Intensity Value");
    ylabel("Frequency");
    title(title_name); 
end

% function to combine segmented images horizontally with thin single pixel
% white line in-between to show segmentation
function combined_segmented_img = combine_segmented_images(images)
    % Loop through the segments
    for i = 1:size(images,2)
        if i == 1
            % initialise a combined image for viewing, put a pixel-thim while
            % line to demarcate segment
            combined_segmented_img = [cell2mat(images(i)) ones(size(cell2mat(images(i)),1), 1)*255];
        elseif i == size(images,2)
            combined_segmented_img = [combined_segmented_img cell2mat(images(i))];
        else
            combined_segmented_img = [combined_segmented_img cell2mat(images(i)) ones(size(cell2mat(images(i)),1), 1)*255];
        end
    end
end

% function to find the root node
function root = Find(node)
    if node.Parent ~= node
        node.Parent = Find(node.Parent);
        root = node.Parent;
    else
        root = node;
    end
end

% function to create union between two nodes (A&B) if they are connected
function Union(node_A, node_B)

    root_A = Find(node_A);
    root_B = Find(node_B);
    
    if root_A == root_B
        return
    end
    
    if root_A.Size < root_B.Size
        temp_A = root_A;
        root_A = root_B;
        root_B = temp_A;
    end
    
    root_B.Parent = root_A;
    root_A.Size = root_A.Size + root_B.Size;
end

% function to return the node from graph_nodes which corresponds to the
% specified label as input
function node = return_node_by_label(specified_label, graph_nodes)
    % loop through each node in graph_nodes
   for count = 1:size(graph_nodes,1)
       % check if the labels match, if so, return the node
       if graph_nodes(count).Label == specified_label
           node = graph_nodes(count);
           return
       end
   end
   % in case the node wsa not found, then the object pixel in question does
   % not associate with any equivalence. It may be an isolated pixel
   node = "no node";
end

% This function determined the centroids of the segmented images
function centroids = determine_centroids(segmented_im, threshold)

    % Create array to store i,j location of the centroids of each image
    centroids = [];

    % identify the centroids of all the segmented images, note that they are
    % likely to be non-integer
    for count = 1:size(segmented_im,2)
        seg_image = cell2mat(segmented_im(count));
        x_arr = [];
        y_arr = [];
        for i = 1:size(seg_image,1)
            for j = 1:size(seg_image,2)
                if seg_image(i,j) >= threshold
                    % store the x and y coordinates
                    x_arr = [x_arr i];
                    y_arr = [y_arr j];
                end
            end
        end
        % find the centroids in the x and y directions and store in centroids
        % array
        centroids = [centroids ; [sum(x_arr)/size(x_arr,2) sum(y_arr)/size(y_arr,2)]];
    end
end

% This function determines the inverse rotated coordinates based on the images and
% centroids as the centre of rotation for backward mapping
function backward_mapping_coordinates = determine_backward_mapping_coordinates(segmented_im, centroids, angle)

    % convert angle in degrees to radians. The angle is negative to signal
    % the inverse transform
    angle_rad = - angle*pi/180;

    % cell array to store the new rotated coordinates for all images for
    % the image transform
    backward_mapping_coordinates = {};

    for count = 1:size(segmented_im,2)
        seg_image = cell2mat(segmented_im(count));
        % create array for new X and Y values
        rotated_coordinates = [];
        for i = 1:size(seg_image,1)
            for j = 1:size(seg_image,2)
                % find displaced x and y coordinates with respect to the
                % centroid of image. i.e. translated coordinates with the
                % centroid as the origin
                displaced_i = centroids(count, 1) - i;
                displaced_j = j - centroids(count, 2);
                % use rotation matrix to find new coordinates with the
                % centroid as the point of rotation in the coordinates with
                % respect to the centroid as the origin
                rotated_i = displaced_i*cos(angle_rad) - displaced_j*sin(angle_rad);
                rotated_j = displaced_i*sin(angle_rad) + displaced_j*cos(angle_rad);
                % translate the corrdinate back to the top left of image
                new_i = centroids(count, 1) - rotated_i;
                new_j = rotated_j + centroids(count, 2);
                % insert the new coordinates back into new_coordinates
                rotated_coordinates = [rotated_coordinates ; [new_i new_j]];
            end
        end
        backward_mapping_coordinates{end+1} = rotated_coordinates;
    end
end

% This function returns segmented images with rotation based on a user
% specified interpolation
function segmented_im_rotated = brightness_interpolation(segmented_im, backward_mapping_coordinates, interpolation_method)

    % create a segmented_im_rotated cell array to store segmented images
    segmented_im_rotated = {};

    for count = 1:size(segmented_im,2)
        % create array for new image and extract old image
        new_image = zeros(size(cell2mat(segmented_im(count)),1),size(cell2mat(segmented_im(count)),2),'uint8');
        old_image = cell2mat(segmented_im(count));
        % counter to loop through the backward_mapping_coordinates for this
        % image
        counter = 0;
        % loop through the i, j coordinates for new image
        for i = 1:size(new_image,1)
            for j = 1:size(new_image,2)
                counter = counter + 1;
                % ensure that backward mapped coordinate is within the
                % image dimensions
                mapped_coordinate = cell2mat(backward_mapping_coordinates(count));
                x_mapped = mapped_coordinate(counter,1);
                y_mapped = mapped_coordinate(counter,2);
                % ensure the mapped coordinates are within the pixel range
                % for the old image
                if x_mapped >= 1 && y_mapped >= 1 && ...
                        x_mapped < size(new_image,1) && y_mapped < size(new_image,2)
                    if interpolation_method == "nearest neighbour"
                        % NN interpolation rounds the values
                        x_interpolated = round(x_mapped);
                        y_interpolated = round(y_mapped);
                        new_image(i, j) = old_image(x_interpolated, y_interpolated);
                    elseif interpolation_method == "bilinear"
                        % bilinear interpolation uses the two dimensional
                        % linear interplation formula. 
                        floor_x = floor(x_mapped);
                        floor_y = floor(y_mapped);
                        a = x_mapped - floor_x;
                        b = y_mapped - floor_y;
                        % implement the formula
                        new_image(i, j) = (1-a)*(1-b)*old_image(floor_x, floor_y)...
                            + a*(1-b)*old_image(floor_x + 1, floor_y)...
                            + (1-a)*b*old_image(floor_x, floor_y + 1)...
                            + a*b*old_image(floor_x + 1, floor_y + 1);
                    elseif interpolation_method == "bicubic"
                        % look at surrounding 16 points and conduct bicubic
                        % interpolation. For this to happen, the range of
                        % valid mapped coordinates are constricted by one
                        % more pixel in each direction. Hence we implement
                        % this additional constraint
                        if x_mapped >= 2 && y_mapped >= 2 && ...
                                x_mapped < size(new_image,1) - 1 && y_mapped < size(new_image,2) - 1
                            floor_x = floor(x_mapped);
                            floor_y = floor(y_mapped);
                            % loop through the 16 points surrounding the
                            % mapped coordinates
                            for x_coord = floor_x - 1 : floor_x + 2
                                for y_coord = floor_y - 1: floor_y + 2
                                    % find the |x| and |y| differences
                                    x_diff = abs(x_mapped - x_coord);
                                    y_diff = abs(y_mapped - y_coord);
                                    % carry our calsulations based on
                                    % values
                                    if x_diff < 1
                                        h_x = 1 - 2*x_diff^2 + x_diff^3;
                                    else
                                       h_x = 4 - 8*x_diff + 5*x_diff^2 - x_diff^3;
                                    end
                                    if y_diff < 1
                                        h_y = 1 - 2*y_diff^2 + y_diff^3;
                                    else
                                       h_y = 4 - 8*y_diff + 5*y_diff^2 - y_diff^3; 
                                    end
                                    new_image(i,j) = new_image(i,j) + h_x*h_y*old_image(x_coord, y_coord);
                                end
                            end
                        end
                    end
                end
            end
        end
        segmented_im_rotated{end+1} = new_image;
    end
end