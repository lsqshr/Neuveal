function singlegt(imgfilename, swcfilename, foreground, dilateradius, blocksize, batchsize, savepath)
% Make the DT transform from the ground truth tracing of a single case
% imgfilename: V3DRaw image file path
% swcfilename: swc file path
% foreground: the foreground threshold for radius estimation
% dM: The expected largest radius

    if ~exist(savepath, 'dir')
        mkdir(savepath);
    end

	[pathstr, ~, ~] = fileparts(mfilename('fullpath'));
	addpath(genpath(fullfile(pathstr, 'FastMarching_version3b')));
    
	% Load V3DRAW
	img = load_v3d_raw_img_file(imgfilename);
	sz = size(img);

	% Load SWC
	swc = load_v3d_swc_file(swcfilename);

	% Use GT to generate a binary map
	bimg = img>foreground;

	% Recalculate the radius of nodes based on the binary map
	for r = 1 : size(swc, 1)
		swc(r, 6) = getradius(bimg, swc(r, 3), swc(r, 4), swc(r, 5));
    end

    swc(swc(r,6) < 2, 6) = 2; % fillin the empty ones
    
	% Make new binary map for the ground truth
	% B = binarysphere3d(size(img), swc(:, 3:5), swc(:, 6));
	[B, pts] = binarycilinder3D(sz, swc); % Returns binary image and the resampled points and their radius
	
	% DT on B
	disp('Distance Transform')
	bdist = getBoundaryDistance(B, true);
    bdistnorm = zeros(size(bdist));

    % Normalise the local area within the radius
    for i = 1 : size(pts, 1) 
        neighbours = neighourpoints3d(pts(i, 1), pts(i, 2), pts(i, 3), pts(i, 4));
        neighbours(:, 1) = constrain(neighbours(:, 1), 1, sz(1));
		neighbours(:, 2) = constrain(neighbours(:, 2), 1, sz(2));
		neighbours(:, 3) = constrain(neighbours(:, 3), 1, sz(3));
		ind = sub2ind(sz, int16(neighbours(:, 1)), int16(neighbours(:, 2)), int16(neighbours(:, 3)));
        d = bdist(ind) / max(bdist(ind));
        bdistnorm(ind) = (d + bdistnorm(ind)) / 2; % if bdist == 0 then it is identical to just assignment 
    end

	bdistnorm(bdistnorm > 1) = 1;
	bdistnorm = exp(bdistnorm) - 1;

%     save(fullfile(savepath, sprintf('B.mat')), 'B');
%     save(fullfile(savepath, sprintf('bdistnorm.mat')), 'bdistnorm');
    
    % Dilate the Binary map for sampling
    [x,y,z] = meshgrid(-dilateradius:dilateradius, ...
                       -dilateradius:dilateradius, ...
                       -dilateradius:dilateradius);
    se = (x/dilateradius).^2 + (y/dilateradius).^2 + (z/dilateradius).^2 <= 1;
    dilateB = imdilate(B, se);
    
    % Pad the dilateB and original image
    padimg = zeros((blocksize-1) + sz(1), (blocksize-1) + sz(2), (blocksize-1) + sz(2));
    padimg((blocksize - 1) / 2 + 1 : (blocksize - 1) / 2 + sz(1),...
    	   (blocksize - 1) / 2 + 1 : (blocksize - 1) / 2 + sz(2),...
    	   (blocksize - 1) / 2 + 1 : (blocksize - 1) / 2 + sz(3)) = img;

    padB = zeros((blocksize-1) + sz(1), (blocksize-1) + sz(2), (blocksize-1) + sz(3));
    padB((blocksize - 1) / 2 + 1 : (blocksize - 1) / 2 + sz(1),...
    	 (blocksize - 1) / 2 + 1 : (blocksize - 1) / 2 + sz(2),...
    	 (blocksize - 1) / 2 + 1 : (blocksize - 1) / 2 + sz(3)) = dilateB;

    paddist = zeros((blocksize-1) + sz(1), (blocksize-1) + sz(2), (blocksize-1) + sz(3));
    paddist((blocksize - 1) / 2 + 1 : (blocksize - 1) / 2 + sz(1),...
    	    (blocksize - 1) / 2 + 1 : (blocksize - 1) / 2 + sz(2),...
    	    (blocksize - 1) / 2 + 1 : (blocksize - 1) / 2 + sz(3)) = bdistnorm;

    % Extract a block for each one of the voxel in dilateB
    fprintf('Collecting blocks for %s with %d voxels to consider\n', imgfilename);
    nvox = sum(padB(:));
    voxidx = find(padB > 0);
    voxidx = voxidx( randperm(numel(voxidx)) );
    coord = zeros(3, batchsize);
    blocks = zeros(blocksize, blocksize, blocksize, batchsize);
    gt = zeros(1, batchsize);
    startidx = 1;
	
    for i = 1 : nvox
    	fprintf('Extracting %d/%d -- %.3f%%\n', i, nvox, 100*i/nvox);
        [x, y, z] = ind2sub(size(padB), voxidx(i));
        blockidx = mod(i, batchsize)+1;

        % The order of saving blocks : <x, y, z, idx>. This order is for loading .mat in torch
        coord(:, blockidx) = [x, y, z];
        blocks(:, :, :, blockidx) = padimg(x - (blocksize - 1)/2 : x + (blocksize - 1)/2, ...
        	                            y - (blocksize - 1)/2 : y + (blocksize - 1)/2, ...
        	                            z - (blocksize - 1)/2 : z + (blocksize - 1)/2);
        gt(:, blockidx) = paddist(x, y, z);
        
        if mod(i, batchsize) == 0 || i == nvox
	        save(fullfile(savepath, sprintf('coord%d-%d.mat', startidx, i)), 'coord');
	        save(fullfile(savepath, sprintf('blocks%d-%d.mat', startidx, i)), 'blocks');
	        save(fullfile(savepath, sprintf('gt%d-%d.mat', startidx, i)), 'gt');
        	startidx = startidx + batchsize;

        	if nvox - i > batchsize
			    coord = zeros(3, batchsize);
			    blocks = zeros(blocksize, blocksize, blocksize, batchsize);
			    gt = zeros(1, batchsize);
			else
			    coord = zeros(3, nvox - i);
			    blocks = zeros(blocksize, blocksize, blocksize, nvox - i);
			    gt = zeros(1, nvox - i);
			end
	    end
    end
    
    save(fullfile(savepath, 'raw.mat'), 'img');
end


function BoundaryDistance = getBoundaryDistance(I,IS3D)
	% Calculate Distance to vessel boundary

	% Set all boundary pixels as fastmarching source-points (distance = 0)
	if(IS3D),S=ones(3,3,3); else S=ones(3,3); end
	B=xor(I,imdilate(I,S));
	ind=find(B(:));
	if(IS3D)
	    [x,y,z]=ind2sub(size(B),ind);
	    SourcePoint=[x(:) y(:) z(:)]';
	else
	    [x,y]=ind2sub(size(B),ind);
	    SourcePoint=[x(:) y(:)]';
	end

	% Calculate Distance to boundarypixels for every voxel in the volume
	SpeedImage=ones(size(I));
	BoundaryDistance = msfm(SpeedImage, SourcePoint, false, true);

	% Mask the result by the binary input image
	BoundaryDistance(~I)=0;
end


function radius = getradius(inputMatrix, input_x, input_y, input_z)
% Calculate the radius of M * N * Z binary matrix at specific location 
% The specific location is defined by corresponding  input_x, input_y, input_z

	[M, N, Z] = size(inputMatrix);
	%sz vector stores the dimensional information 
	sz(1) = M;
	sz(2) = N;
	sz(3) = Z;
	%max_r is max radius radius which each neuron can have
	max_r = max([M/2, N/2, Z/2]);

	% mx, my, mz are eight neighbouring points of center points
	mx = input_x + 0.5;
	my = input_y + 0.5;
	mz = input_z + 0.5;

	%tol_num is the total number of foreground voxels
	%bak_num is the total number of background voxels
	tol_num = 0;
	bak_num = 0;

	%neighpot represent eight choosing ways. 
	%It either choose the first element or the second element.
	neighpot = [2, 1, 1;
				1, 2, 1;
				1, 1, 2;
				2, 1, 2;
				2, 2, 1;
				1, 2, 2;
				2, 2, 2;
				2, 1, 1;];
	for(r = 1 : max_r)
		r1 = r - 0.5;
		r2 = r + 0.5;
		r1_r1 =  r1 * r1;
		r2_r2 =  r2 * r2;
		z_min = 0;
		z_max = r2;
		for(dz = z_min : 1 : z_max) 
			dz_dz = dz * dz;
			y_min = 0;
			y_max =  sqrt(r2_r2 - dz_dz);
			for (dy = y_min : y_max)
				dy_dy = dy * dy;
				x_min = r1_r1 - dz_dz - dy_dy;
				if (x_min > 0)
					x_min = sqrt(x_min) + 1;
				else
					x_min = 0;
				end
	            x_max = sqrt(r2_r2 - dz_dz - dy_dy);
				for (dx = x_min : 1 : x_max)
					x(1) = mx - dx;
					x(2) = mx + dx;
					y(1) = my - dy;
					y(2) = my + dy;
					z(1) = mz - dz;
					z(2) = mz + dz;
					for (b = 1 : 8)
						neighindex = neighpot(b, :, :);
						ii = neighindex(1);
						jj = neighindex(2);
						kk = neighindex(3);
						%Make sure that center point is still in the 3D binary matrix
						if (x(ii)<1 || x(ii)>sz(1) || y(jj)<1 || y(jj)>sz(2) || z(kk)<1 || z(kk)>sz(3))
							%radius is final return value 
							radius = r;
							return;
						else
							tol_num = tol_num + 1;
							point = inputMatrix(round(x(ii)), round(y(jj)), round(z(kk)));
							%fprintf('bak_num : %6.2f  tol_num: %12.8f\n',bak_num, tol_num);
	                        if(point == 0)
								bak_num = bak_num + 1;
	                        end
	                        %This is the criterion to calculate the radius, I have to say it is a harsh criterion which might be adjusted in the future 
	                        if ((bak_num / tol_num) > 0.4)
								radius = r;
	                            return;
							end
						end
					end
				end
			end
		end
	end
	radius = r;
	return

end


function [bcilinder, pts] = binarycilinder3D(sz, swc)
% Generate binary image by the swc skelonton
% Draw cilinder shape between each pair of adjacent nodes.

	bcilinder = logical(zeros(sz));

    % figure(1)
    % axis([0 sz(1) 0 sz(2) 0 sz(3)]);
    % hold on

    pts = [];
	for i = 1 : size(swc, 1)
		node = swc(i, :);

	    if swc(i, 7) > 0 
	    	pid = swc(i, 7);
	    	pnode = swc(swc(:,1) == pid, :);

            if ~any(swc(:,1) == pid) || ...
               any(pnode(3:5) < 0) || ...
               any(pnode(3:5) > sz) || ...
               any(node(3:5) < 0) || ...
               any(node(3:5) > sz)
                continue
            end

            D = sqrt(sum((pnode(3:5) - node(3:5)) .^ 2 ));
            steps = floor(D) + 2;
            stepsize = (pnode(3:5) - node(3:5)) / (steps - 1);
            dr = (pnode(:, 6) - node(:, 6)) / (steps - 1);
            r = node(6);

            for j = 0 : steps - 1
                node(3:5) = node(3:5) + j * (stepsize);
                % plot3(node(3), node(4), node(5), 'r*');
				neighbours = neighourpoints3d(node(3), node(4), node(5), floor(r + j * dr));
				neighbours(:, 1) = constrain(neighbours(:, 1), 1, sz(1));
				neighbours(:, 2) = constrain(neighbours(:, 2), 1, sz(2));
				neighbours(:, 3) = constrain(neighbours(:, 3), 1, sz(3));
				ind = sub2ind(sz, int16(neighbours(:, 1)), int16(neighbours(:, 2)), int16(neighbours(:, 3)));
				bcilinder(ind) = 1;
				pts = [pts; node(3), node(4), node(5), floor(r + j * dr)];
            end
	    end
	end
	% hold off
end


function neighours = neighourpoints3d(x, y, z, radius)
% Return the coordinates of neighours within a radius
	xgv = [(x - radius) : (x + radius)];
	ygv = [(y - radius) : (y + radius)];
	zgv = [(z - radius) : (z + radius)];
	[x, y, z] = meshgrid(xgv, ygv, zgv); % Rectangular grid in 2-D
	neighours = [x(:), y(:), z(:)];

end


function x = constrain(x, a, b)
	% constrain a vector between a and b
	x(x<a) = a;
	x(x>b) = b;
end