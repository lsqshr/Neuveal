function visualiseblocks(blockpath, gtpath, nsample)
	f = load(blockpath);
	blks = f.blocks;
    
    f = load(gtpath);
    gt = f.gt;
    
    randidx = randi(size(blks, 4), nsample, 1);
    sampledblks = blks(:, :, :, randidx);
    sampledgt = gt(randidx);
    
    figure(1)
    ncol = floor(nsample .^ 0.5);
    if nsample == ncol ^ 2
    	nrow = ncol;
    else
    	nrow = ncol + 1;
    end

    for r = 1 : nrow
    	for c = 1 : ncol
    		fidx = c + (r - 1) * ncol;
    		if fidx > nsample
    			break;
            end
            
    		subplot(nrow, ncol, fidx);
    		imagesc(max(sampledblks(:,:,:,fidx), [], 3));
            title(sprintf('%d', sampledgt(fidx)));
    	end
    end

end