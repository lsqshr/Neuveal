OPPATH = '/home/siqi/hpc-data1/OP_V3Draw/OP_V3Draw-block';
ctr = 0;
imgblock = {};
synblock = {};

for i = 1 : 9
	if i ~= 2
		filepath = fullfile(OPPATH, sprintf('OP_%d-block.mat', i));
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

imgblockmat = imgblockmat ./ max(imgblockmat(:));
synblockmat = synblockmat ./ max(synblockmat(:));

save(fullfile(OPPATH, 'whole-op.mat'), 'imgblockmat', 'synblockmat');
hdf5write(fullfile(OPPATH, 'whole-op-img.h5'), '/op/img', uint8(imgblockmat));
hdf5write(fullfile(OPPATH, 'whole-op-syn.h5'), '/op/syn', uint8(synblockmat));