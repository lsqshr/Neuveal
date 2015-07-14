function dataprep(datapath, gtpath, radius)
% Main Script for Preparing the data for training NeuDraw

SAVESINGLEBLOCK = true; 
YINVERSE = true;
DEBUG = false;

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

synpath = fullfile(datapath, [datadirname, '-syn']);
if ~exist(synpath)
   mkdir(synpath);
end

blockpath = fullfile(datapath, [datadirname, '-block']);
if ~exist(blockpath)
   mkdir(blockpath);
end

fprintf('found %d v3draw in %s\n', numel(ldatapath), datapath);

for i = 1 : length(ldatapath)
    fname = ldatapath(i).name;
    srcpath = fullfile(datapath, fname);
    fprintf('Loading Image from: %s\n', srcpath);
    img = load_v3d_raw_img_file(srcpath);
    if numel(size(img)) > 3
        img = squeeze(img(:, : , :, 1));
    end
    [croped, cropregion]  = imagecrop(img, 0);
    [~, fname, ~] = fileparts(fname);

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
    synimg = synthesize_gtimage(swc, imgsz, radius);
    fout = fullfile(synpath, [fname '-syn.v3draw']);
    fprintf('Saving synthesized image to: %s\n', fout);
    save_v3d_raw_img_file(synimg, fout);

    % Extract 3D blocks from both the original images and the synthesized image
    [imgblock, synblock] = extractblock(swc, croped, synimg, 30);
    imgblock = uint8(imgblock);
    synblock = uint8(synblock);
    save(fullfile(blockpath, [fname, '-block.mat']), 'imgblock', 'synblock');

    % Save single blocks to v3d for visual check 
    if SAVESINGLEBLOCK
        for b = 1 : size(imgblock, 1)
            if ~exist(fullfile(blockpath, [fname '-block']))
                mkdir(fullfile(blockpath, [fname '-block']))
            end

            oriimgblkpath = fullfile(blockpath, [fname '-block'], [fname '-ori-block-' num2str(b) '.v3draw']);
            fprintf('Saving ori imgblock to %s', oriimgblkpath);
            save_v3d_raw_img_file(squeeze(imgblock(b, :, :, :)), oriimgblkpath)

            synimgblkpath = fullfile(blockpath, [fname '-block'], [fname '-gt-block-' num2str(b) '.v3draw']);
            fprintf('Saving gt imgblock to %s', synimgblkpath);
            save_v3d_raw_img_file(squeeze(synblock(b, :, :, :)), synimgblkpath)
        end
    end

    if DEBUG
        break;
    end 
end

end


function [imgblock, synblock] = extractblock(swc, img, synimg, blocksize)
    blockradius = floor(blocksize / 2);

    % Padding both images with 2 * block size
    imgpad = zeros(size(img, 1) + 4 * blocksize, size(img, 2) + ...
                   4 * blocksize, size(img, 3) + 4 * blocksize);
    synpad = zeros(size(synimg, 1) + 4 * blocksize, size(synimg, 2) + ...
                   4 * blocksize, size(synimg, 3) + 4 * blocksize);
    % Shift all nodes to padding
    swc(:, 3:5) = swc(:, 3:5) + 2 * blocksize;

    imgpad(2 * blocksize + 1 : end - 2 * blocksize,...
           2 * blocksize + 1 : end - 2 * blocksize,...
           2 * blocksize + 1 : end - 2 * blocksize) = img;
    synpad(2 * blocksize + 1 : end - 2 * blocksize,...
           2 * blocksize + 1 : end - 2 * blocksize,...
           2 * blocksize + 1 : end - 2 * blocksize) = synimg;

    blocksize = blockradius * 2 + 1;
    imgblock = zeros(size(swc, 1), blocksize, blocksize, blocksize);
    synblock = zeros(size(swc, 1), blocksize, blocksize, blocksize);

    % For each point in swc take a block with a randomshift which is smaller than blocksize
    for i = 1 : size(swc, 1)
        x = swc(i,3);
        y = swc(i,4);
        z = swc(i,5);
        % shiftx = randi(blockradius + 1) - 1 - 0.25 * blocksize;
        % shifty = randi(blockradius + 1) - 1 - 0.25 * blocksize;
        % shiftz = randi(blockradius + 1) - 1 - 0.25 * blocksize;
        % x = floor(x + shiftx);
        % y = floor(y + shifty);
        % z = floor(z + shiftz);

        imgblock(i, :, : , :) = imgpad(x - blockradius : x + blockradius,...
                                       y - blockradius : y + blockradius,...
                                       z - blockradius : z + blockradius);
        synblock(i, : , :, :) = synpad(x - blockradius : x + blockradius,...
                                       y - blockradius : y + blockradius,...
                                       z - blockradius : z + blockradius);
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


function synimg = synthesize_gtimage(swc, sz, radius)
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

% Combine OPs
% TODO: Should change to fit general cases
ctr = 0;
imgblock = {};
synblock = {};

for i = 1 : 9
	if i ~= 2
		filepath = fullfile(blockpath, sprintf('OP_%d-block.mat', i));
		f = load(filepath);
		imgblock{i} = f.imgblock;
		synblock{i} = f.synblock;
		ctr = ctr + size(f.imgblock, 1);
	end
end

imgblockmat = zeros(ctr, 31, 31, 31);
synblockmat = zeros(ctr, 31, 31, 31);
matctr = 1;

for i = 1 : 9
	if i ~= 2
		imgblockmat(matctr : size(imgblock{i}, 1), :, :, :) = imgblock{i};
		synblockmat(matctr : size(synblock{i}, 1), :, :, :) = synblock{i};
	end
end

save(fullfile(OPPATH, 'whole-op.mat'), 'imgblockmat', 'synblockmat');
hdf5write(fullfile(OPPATH, 'whole-op.h5'), '/img', uint8(imgblockmat));
hdf5write(fullfile(OPPATH, 'whole-op.h5'), '/syn', uint8(synblockmat));