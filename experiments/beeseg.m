
try
    v = VideoReader('C:\Users\dom\Downloads\Videos\1.tracking\count-hard.mp4');
    p = vision.VideoPlayer;
catch ex
    disp(['Error reading video ' filename '. Reason: ' ex.message]);
    return;
end

every_nth_frame = 1;

detector = vision.ForegroundDetector('NumGaussians', 3, 'NumTrainingFrames', 15);
   
blob = vision.BlobAnalysis(...
   'CentroidOutputPort', false, 'AreaOutputPort', false, ...
   'BoundingBoxOutputPort', true, ...
   'MinimumBlobAreaSource', 'Property', 'MinimumBlobArea', 250);

shapeInserter = vision.ShapeInserter('BorderColor','White');

for c = 1:every_nth_frame:v.NumberOfFrames - 1 % 
    frame = read(v, c);
    frame = imresize(frame, 0.2);
    frame = im2double(frame);
    
    hsv = rgb2hsv(frame);
    %hsv_mask = zeros(size(frame));
    hsv_mask = hsv(:,:,2) > 0.6;
    
    %frame2 = frame;
    %frame2(:,:,1) = 0;
    %frame2(:,:,2) = 0;
    
    frame_dbl = frame;
    mask = frame_dbl(:,:,3) < 0.2;
    %out = repmat(mask, [1 1 3]);
    
    [yRed, x] = imhist(frame(:,:,1));
    [yGreen, x] = imhist(frame(:,:,2));
    [yBlue, x] = imhist(frame(:,:,3));
    plot(x, yRed, 'Red', x, yGreen, 'Green', x, yBlue, 'Blue');
    
    fgframe= rgb2gray(frame);
    fgMask = step(detector, frame);
    
    % combine masks
    mask = fgMask | hsv_mask;
    se = strel('square', 5);
    filtered_mask = imopen(mask, se);
    
    bbox   = step(blob, fgMask);
    out    = step(shapeInserter, frame, bbox);
    
    out(out(:,:,3) < 0.1) = 0;
    
    step(p, [frame repmat(hsv(:,:,2), [1 1 3])] );
end