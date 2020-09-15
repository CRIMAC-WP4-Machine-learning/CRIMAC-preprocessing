function CRIMAC_preprocess(survey,dd_data_in,dd_data_work,dd_data_out,plot_frequency,plt)
% This script reads the metadata, raw acoustic data and the labels, convert the 
% input data to a mat file that include both raw data and labels. The script
% also interpolates the data into a common grid.
%
% Dependencies:
% https://github.com/nilsolav/LSSSreader/src
% https://github.com/nilsolav/NMDAPIreader
% https://github.com/nilsolav/readEKraw
%

% CRIMAC project, Nils Olav Handegard
if nargin<5
    plot_frequency=200;
    plt=false;
end
% Plotting frequency
% Which frequency to use when generating the plots
par.plot_frequency=num2str(plot_frequency);

% Which range vector to use when interpolating into the common grid
par.range_frequency = plot_frequency;

par.bottomoutlier = 95; % Assumes less than 5 percent outliers in depthdata
par.depthoffset = 15;% Add par.depthoffset m below seafloor
par.dz = 0.188799999523326;
par.dzdiff = .05;

% Survey data directory per year
%% TEMPORARY for testing purposes
if nargin == 0
  if isunix
    dd_data_in = '/datain/';
    dd_data_work = '/datawork/';
    dd_data_out = '/dataout/';
    survey = ''
  else % For testing purposes on PC
    dd_data_in = 'D:\DATA\LSSS-label-versioning\S2017838\ACOUSTIC\EK60\EK60_RAWDATA';
    dd_data_work = 'D:\DATA\LSSS-label-versioning\S2017838\ACOUSTIC\LSSS\WORK';
    dd_data_out = 'D:\DATA\LSSS-label-versioning\S2017838\ACOUSTIC\memmap';
    survey = 'S2017838'
  end
end

%% \TEMPORARY
    
% Get the file list
raw0 = dir(fullfile(dd_data_in,'*.raw'));
    
% Generate status file if it is missing
statusfile = fullfile(dd_data_out,[survey,'datastatus.mat']);

if ~exist(statusfile)
    status = zeros(length(raw0),1);
    save(statusfile,'status')
end

% Loop over file pairs
for f=1:length(raw0)
    load(statusfile)
    % Create file names (in and out)
    [~,fn,~]=fileparts(raw0(f).name);
    % Run files that have not been run earlier
    qrun = status(f)<=0;
    % Get files
    bot = fullfile(dd_data_in,[fn,'.bot']);
    raw = fullfile(dd_data_in,[fn,'.raw']);
    snap = fullfile(dd_data_work,[fn,'.work']);
    % Output files
    mat = fullfile(dd_data_out,[fn,'.mat']);
    if plt
        png = fullfile(dd_data_out,[fn,'.png']);
        png_I = fullfile(dd_data_out,[fn,'_I.png']);
        png_I2 = fullfile(dd_data_out,[fn,'_I2.png']);
    else
        png=[];
        png_I =[];
        png_I2 =[];
    end
    
    if qrun
        disp([datestr(now),'; running ; ',fullfile(dd_data_out,fn)])
        % Generate figures and save clean data file
        if ~exist(snap,'file')
          disp('No interpretation file')
          snap=[];
        end
        fexist(raw)
        fexist(bot)
        try
            CRIMAC_preprocess_generate_mat_files(snap,raw,bot,png,png_I,png_I2,mat,par)
            %close gcf
            disp([datestr(now),'; success ; ',fn])
            status(f)=now;
        catch ME
            disp([datestr(now),'; failed  ; ',fn])
            status(f)=-now;
            disp([ME.identifier]);
            disp([ME.message]);
            for MEi = 1:length(ME.stack)
                disp(ME.stack(MEi))
            end
        end
    else
        disp([datestr(now),'; exists ; ',fn])
    end
    save(statusfile,'status')
end

function fexist(file)
if ~exist(file,'file')
    error(['Missing file:',file])
end

