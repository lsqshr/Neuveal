function dataprep(datapath, gtpath)
% Main Script for Preparing the data for training NeuDraw
curdir = fileparts(mfilename('fullpath'));
[dataparentdir, datadirname] = fileparts(datapath);
outpath = fullfile(dataparentdir, [datadirname, '-cropped']);
addpath(fullfile(curdir, '..', '..', ...
	'v3d-compiled/v3d_external/matlab_io_basicdatatype'));
ldatapath = dir([datapath, [filesep, '*.v3draw']]);

if ~exist(outpath)
   mkdir(outpath);
end

for i = 1 : length(ldatapath)
    fname = ldatapath(i).name;
    srcpath = fullfile(datapath, fname);
    fprintf('Loading Image from: %s', srcpath);
    img = load_v3d_raw_img_file(srcpath);
    croped = imagecrop(img, 0);
    [~, fname, ~] = fileparts(fname);
    fout = fullfile(outpath, [fname '-croped.v3draw'])
    fprintf('Saving to: %s', fout);
    save_v3d_raw_img_file(croped, fout);
end

end