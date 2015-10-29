function dataprep(datapath, gtpath, v3dpath, radius)
% Main Script for Preparing the data for training NeuDraw
    addpath(v3dpath);
    SAVESINGLEBLOCK = false; 
    YINVERSE = true;
    DEBUG = false;
    blocksize = [30, 30, 20];
    syndownsample = [28, 31, 16];

    curdir = fileparts(mfilename('fullpath'));
    [dataparentdir, datadirname] = fileparts(datapath);
    addpath(fullfile(curdir, '..', '..', ...
    	'vaa3d_matlab_io'));
    ldatapath = dir([datapath, [filesep, '*.v3draw']]);

    croppath = fullfile(datapath, [datadirname, '-cropped']);
    if ~exist(croppath)
       mkdir(croppath);
    end

    cropgtpath = fullfile(datapath, [datadirname, '-croped-gt']);
    if ~exist(cropgtpath)
       mkdir(cropgtpath);
    end

    inversegtpath = fullfile(datapath, [datadirname, '-inversed-gt']);
    if ~exist(cropgtpath)
       mkdir(cropgtpath);
    end

    synskeletonpath = fullfile(datapath, [datadirname, '-syn-skeleton']);
    if ~exist(synskeletonpath)
       mkdir(synskeletonpath);
    end

    blockpath = fullfile(datapath, [datadirname, '-block']);
    if ~exist(blockpath)
       mkdir(blockpath);
    end

    fprintf('found %d v3draw in %s\n', numel(ldatapath), datapath);

    sbjctr = 0;
    for i = 1 : length(ldatapath)
        fname = ldatapath(i).name;
        srcpath = fullfile(datapath, fname);
        fprintf('Loading Image from: %s\n', srcpath);
        img = load_v3d_raw_img_file(srcpath);
        if numel(size(img)) > 3
            img = squeeze(img(:, : , :, 1));
        end
        [croped, cropregion]  = imagecrop(img, 0.1);
        [~, fname, ~] = fileparts(fname);

        % Add poisson and S&P noises to the cropped image
        noised = imnoise(croped, 'poisson');
        noised = imnoise(croped, 'salt & pepper', 0.01); % Might be problem

        fout = fullfile(croppath, [fname '-croped.v3draw'])
        fprintf('Saving cropped image to: %s\n', fout);
        save_v3d_raw_img_file(croped, fout);

        % Read in the swc tree in tree model
        fgt = fullfile(gtpath, [fname '.swc']); % Assume the ground truth is named as the datafile + '.swc'
        swc = load_v3d_swc_file(fgt);
        assert(size(swc, 1) ~= 0 );
        
        if YINVERSE
            ycoor = swc(:, 4);
            swc(:, 4) = abs(ycoor - size(img, 2) - 1);
        end

        fprintf('Saving inversed gt to: %s\n', fout);
        fout = fullfile(cropgtpath, [fname '-inversed.swc']);
        save_v3d_swc_file(swc, fout);

        % Transfer all swc node coordinate according to the crop region
        swc(:, 3) = swc(:, 3) - cropregion(1, 1); % x
        swc(:, 4) = swc(:, 4) - cropregion(2, 1); % y
        swc(:, 5) = swc(:, 5) - cropregion(3, 1); % z
        fprintf('Saving transfered gt to: %s\n', fout);
        fout = fullfile(cropgtpath, [fname '-croped.swc']);
        save_v3d_swc_file(swc, fout);

        % Draw the sythesized ground truth image
        imgsz = cropregion(:, 2) - cropregion(:, 1);
        synimg = synthesize_skeleton(swc, imgsz, radius);
        fout = fullfile(synskeletonpath, [fname '-syn.v3draw']);
        fprintf('Saving synthesized image to: %s\n', fout);
        save_v3d_raw_img_file(synimg, fout);

        % Extract 3D blocks from both the original images and the synthesized image
        [imgblock, blocksize] = extractblock(swc, croped, blocksize);
        [synblock, blocksize] = extractblock(swc, synimg, blocksize);
        [noisedblock, blocksize] = extractblock(swc, noised, blocksize);

        imgblock = uint8(imgblock);
        synblock = uint8(synblock);
        noisedblock = uint8(noisedblock);
        save(fullfile(blockpath, [fname, '-block.mat']), 'imgblock', 'synblock', 'noisedblock');

        % Save single blocks to v3d for visual check 
        if SAVESINGLEBLOCK
            for b = 1 : size(imgblock, 1)
                if ~exist(fullfile(blockpath, [fname '-block']))
                    mkdir(fullfile(blockpath, [fname '-block']))
                end

                oriimgblkpath = fullfile(blockpath, [fname '-block'], [fname '-ori-block-' num2str(b) '.v3draw']);
                save_v3d_raw_img_file(squeeze(imgblock(b, :, :, :)), oriimgblkpath)

                synimgblkpath = fullfile(blockpath, [fname '-block'], [fname '-gt-block-' num2str(b) '.v3draw']);
                save_v3d_raw_img_file(squeeze(synblock(b, :, :, :)), synimgblkpath)


                noisedblkpath = fullfile(blockpath, [fname '-block'], [fname '-noise-block-' num2str(b) '.v3draw']);
                save_v3d_raw_img_file(squeeze(noisedblock(b, :, :, :)), noisedblkpath)

                if b > 20 && DEBUG
                    break;
                end 
            end
        end

        sbjctr = sbjctr + 1; 

        if DEBUG
            break;
        end 

    end

    % Combine OPs
    ctr = 0;
    imgblock = {};
    synblock = {};
    noisedblock = {};

    for i = 1 : sbjctr 
    	if i ~= 2
    		filepath = fullfile(blockpath, sprintf('OP_%d-block.mat', i));
    		f = load(filepath);
    		imgblock{i} = f.imgblock;
    		synblock{i} = f.synblock;
            noisedblock{i} = f.noisedblock;
    		ctr = ctr + size(f.imgblock, 1);
    	end
    end

    noisedblockmat = zeros(ctr, blocksize(1), blocksize(2), blocksize(3));
    imgblockmat = zeros(ctr, blocksize(1), blocksize(2), blocksize(3));
    synblockmat = zeros(ctr, blocksize(1), blocksize(2), blocksize(3));
    matctr = 1;

    for i = 1 : sbjctr 
        assert(size(imgblock{i}, 1) == size(synblock{i}, 1));
    	if i ~= 2
    		imgblockmat(matctr : matctr + size(imgblock{i}, 1) - 1, :, :, :) = imgblock{i};
    		synblockmat(matctr : matctr + size(synblock{i}, 1) - 1, :, :, :) = synblock{i};
            noisedblockmat(matctr : matctr + size(noisedblock{i}, 1) - 1, :, :, :) = noisedblock{i};
    	end
        
        matctr = matctr + size(imgblock{i}, 1);
    end

    % Downsample the synblocks to the outputsize of the dcnn
    synblockmat_downsample = zeros(size(synblockmat, 1),...
                                   syndownsample(1), syndownsample(2),...
                                   syndownsample(3));
    imgblockmat_downsample = zeros(size(imgblockmat, 1),...
                                   syndownsample(1), syndownsample(2),...
                                   syndownsample(3));
    
    for i = 1 : size(synblockmat, 1)
        im = squeeze(synblockmat(i, :, :, :));
        [y x z]= ndgrid(linspace(1, size(im, 1), syndownsample(1)),...
                        linspace(1, size(im, 2), syndownsample(2)),...
                        linspace(1, size(im, 3), syndownsample(3)));
        synblockmat_downsample(i, :, :, :) = interp3(im, x, y, z, 'spline');

        % Downsample original block (croped)
        im = squeeze(imgblockmat(i, :, :, :));
        [y x z]= ndgrid(linspace(1, size(im, 1), syndownsample(1)),...
                        linspace(1, size(im, 2), syndownsample(2)),...
                        linspace(1, size(im, 3), syndownsample(3)));
        imgblockmat_downsample(i, :, :, :) = interp3(im, x, y, z, 'spline');

        if SAVESINGLEBLOCK
            synimgblkpath = fullfile(blockpath, [fname '-block'], [fname '-gt-block-down-' num2str(i) '.v3draw']);
            save_v3d_raw_img_file(uint8(squeeze(synblockmat_downsample(i, :, :, :))), synimgblkpath);
            imgblkpath = fullfile(blockpath, [fname '-block'], [fname '-ori-block-down-' num2str(i) '.v3draw']);
            save_v3d_raw_img_file(uint8(squeeze(imgblockmat_downsample(i, :, :, :))), imgblkpath);
        end
    end
    
    imgblockmat = imgblockmat ./ max(imgblockmat(:));
    noisedblockmat = noisedblockmat ./ max(noisedblockmat(:));
    synblockmat = synblockmat ./ max(synblockmat(:));
    synblockmat_downsample = synblockmat_downsample ./ max(synblockmat_downsample(:));
    imgblockmat_downsample = imgblockmat_downsample ./ max(imgblockmat_downsample(:));

    save(fullfile(datapath, 'whole-op.mat'), 'imgblockmat', 'synblockmat', 'noisedblockmat', 'synblockmat_downsample');
    hdf5write(fullfile(datapath, 'whole-op-img.h5'), '/op/img', imgblockmat);
    hdf5write(fullfile(datapath, 'whole-op-syn.h5'), '/op/img', synblockmat);
    hdf5write(fullfile(datapath, 'whole-op-noise.h5'), '/op/img', noisedblockmat);
    hdf5write(fullfile(datapath, 'whole-op-img-downsample.h5'), '/op/img', imgblockmat_downsample);
    hdf5write(fullfile(datapath, 'whole-op-syn-downsample.h5'), '/op/img', synblockmat_downsample);
end


function [imgblock, blocksize] = extractblock(swc, img, blocksize)
    blockradius = floor(blocksize / 2);

    % Padding both images with 2 * block size
    imgpad = zeros(size(img, 1) + 4 * blocksize(1), size(img, 2) + ...
                   4 * blocksize(2), size(img, 3) + 4 * blocksize(3));
    % Shift all nodes to padding
    swc(:, 3) = swc(:, 3) + 2 * blocksize(1);
    swc(:, 4) = swc(:, 4) + 2 * blocksize(2);
    swc(:, 5) = swc(:, 5) + 2 * blocksize(3);

    imgpad(2 * blocksize(1) + 1 : end - 2 * blocksize(1),...
           2 * blocksize(2) + 1 : end - 2 * blocksize(2),...
           2 * blocksize(3) + 1 : end - 2 * blocksize(3)) = img;

    blocksize = blockradius * 2 + 1;
    imgblock = zeros(size(swc, 1), blocksize(1), blocksize(2), blocksize(3));

    % For each point in swc take a block with a randomshift which is smaller than blocksize
    for i = 1 : size(swc, 1)
        x = swc(i,3);
        y = swc(i,4);
        z = swc(i,5);

        imgblock(i, :, : , :) = imgpad(x - blockradius(1) : x + blockradius(1),...
                                       y - blockradius(2) : y + blockradius(2),...
                                       z - blockradius(3) : z + blockradius(3));
    end

end


function [croped, cropregion] = imagecrop(srcimg, threshold)
	srcimg = squeeze(srcimg);
    ind = find(srcimg > threshold);
    [M, N, Z] = ind2sub(size(srcimg), ind);
    cropregion = [min(M), max(M); min(N), max(N); min(Z), max(Z)];
    croped = srcimg(cropregion(1, 1) : cropregion(1, 2), ...
                    cropregion(2, 1) : cropregion(2, 2), ...
                    cropregion(3, 1) : cropregion(3, 2));
end


function synimg = synthesize_skeleton(swc, sz, radius)
	synimg = zeros(sz(1), sz(2), sz(3));
	nonsourceidx = swc(:, end) ~= -1;

    % Draw a line on the synthesized image for each edge
    % The order does not matter
    edges = [swc(nonsourceidx, 3:5) swc( swc(nonsourceidx, end), 3:5)];

    for i = 1 : size(edges, 1)
        dist = ceil(((edges(i, 1) - edges(i, 4))^2 + ...
                     (edges(i, 2) - edges(i, 5))^2 + ...
                     (edges(i, 3) - edges(i, 6))^2)^0.5);
    	lx = ceil(linspace(edges(i, 1), edges(i, 4), dist * 2));
    	ly = ceil(linspace(edges(i, 2), edges(i, 5), dist * 2));
    	lz = ceil(linspace(edges(i, 3), edges(i, 6), dist * 2));
        lx(lx<1) = 1;
        ly(ly<1) = 1;
        lz(lz<1) = 1;

    	% Draw line on synimg
    	synimg(lx, ly, lz) = 255 * 7;
    end

    rng(9, 'twister');
    synimg = smooth3(synimg, 'gaussian', 5, 1.2);

    synimg = uint8(synimg);

end


% function synthesize_deconv(srcimg, swc, d)
% % Boost the intensity of the pixels close to the skeleton dist(x, y, z, swc) < d
% % Penalize the intensity of the pixels far from the skeleton dist(x, y, z, swc) > d

%     gtpos = swc(:, 3:5);
%     ngt = size(gtpos, 1)
%     % Find coordinates with intensity > 0.1
%     ind = find(srcimg > 0.1);
%     sub = ind2sub(size(srcimg), ind);
%     lmindist = zeros(size(sub, 1), 1);
%     weight = zeros(size(sub, 1), 1);
%     % Calculate the distances between each coordinate and all groundtruth point
%     % Using forloop in consideration of memory usage
%     for i = 1 : size(sub, 1)
%         subrep = repmat(sub(i, :), ngt, 1);
%         dist = (sum((subrep - gtpos).^2, 2)) .^ 0.5;
%         lmindist(i) = min(dist);
%     end

%     % Use the min distance to calculate the leverage weight
%     weight(lmindist > d) = -1 * 
% end
