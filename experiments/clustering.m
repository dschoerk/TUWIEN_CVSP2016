
hsv = rgb2hsv(frame);
ab = double(hsv(:,:,[2 3]));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols, 2);

scatter(ab(:,1), ab(:,2), 1);
figure;
imshow(hsv .* repmat(hsv(:,:,1) < 0.3, [1 1 3]) .* repmat(hsv(:,:,2) > 0.55, [1 1 3]));
figure;
imshow(hsv .* repmat(hsv(:,:,2) > 0.55, [1 1 3]));