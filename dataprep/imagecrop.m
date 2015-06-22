function croped = imagecrop(srcimg, threshold)
	srcimg = squeeze(srcimg);
    ind = find(srcimg > threshold);
    [M, N, Z] = ind2sub(size(srcimg), ind);
    croped = srcimg(min(M):max(M), min(N):max(N), min(Z):max(Z));
end
