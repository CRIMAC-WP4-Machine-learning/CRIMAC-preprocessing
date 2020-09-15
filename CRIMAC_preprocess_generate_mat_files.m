function CRIMAC_preprocess_generate_mat_files(snap,raw,bot,png,png_I,png_I2,datfile,par)
% function CM_AC_createimages(snap,raw,png,datfile,par)
% 
% This function reads the snap and raw files and generates
% an image per file and a mat file containing the data for
% furhter processing in python.
%
% If there are mising pings in a frequency, the ping is replaced
% with NaN. 
%
% If the range vectors are different between frequencies the 
% sv values are resampled or averaged onto the range vector
% for the frequency in par.range_frequency
% 
% Input:
% snap : Full path snap file
% raw  : Full path to the raw file
% png  : Full path to the image file (can be empty)
% datfile    : full path to the mat output file
% par.plot_frequency      : Plot frequency
% par.range_frequency rangeF : The frequency for the range resolution on the final data
%

% COGMAR project, Nils Olav Handegard
plt=false;
f = par.plot_frequency;

% Read snap file
if ~isempty(snap)
[school,layer,exclude,erased] = LSSSreader_readsnapfiles(snap);
end

% Read raw file and convert to data
[raw_header,raw_data] = readEKRaw(raw,'Heave','True');
raw_cal = readEKRaw_GetCalParms(raw_header, raw_data);
data = readEKRaw_Power2Sv(raw_data,raw_cal,'Linear',true);

% Get depth information
[~, bottom, ~] = readEKBot(bot,raw_cal,raw_data);
% Get the main frequency
for ch = 1:length(raw_data.pings)
    F(ch)=raw_data.pings(ch).frequency(1)/1000;
end

% Sometimes there are missing frequencies, if so, give error
br = false;
if isempty(find(F==(str2num(f))))
    disp('Missing main plotting frequency')
    br=true;
end
if isempty(find(F==par.range_frequency))
    disp('Missing main range frequency')
    br=true;
end
if br
    error('Missing frequencies')
end

%% Maximum range (remove the data below sea floor)
maxrange = prctile(bottom.pings.bottomdepth(:),par.bottomoutlier)+par.depthoffset; % Assumes less than 2 percent outliers

%% Plot result
if ~isempty(png)
    ch = find(F==(str2num(f)));
    rangeind = data.pings(ch).range < maxrange;
    td = double(median(raw_data.pings(ch).transducerdepth));
    [fh, ih] = readEKRaw_SimpleEchogram(10*log10(data.pings(ch).sv(rangeind,:)), 1:length(data.pings(ch).time), data.pings(ch).range(rangeind));
    % Plot the interpretation mask
    hold on
    if ~isempty(snap)
        LSSSreader_plotsnapfiles(layer,school,erased,exclude,f,ch,td)
    end
    plot(bottom.pings.bottomdepth(ch,:));
    title([f,'kHz'])
    print(png,'-dpng')
    close(gcf)
end

%% Extract clean data file
if ~isempty(datfile)
    %% Reshape data
    % Find the main frequency
    Fi=find(F==par.range_frequency);
    
    heave = raw_data.pings(Fi).heave;
    trdepth = raw_data.pings(Fi).transducerdepth;
    
    % Find the ping interval for the main frequency
    % Occasionally there are a negative time diff
    % in the raw data, hence the 'abs' operator.
    tol = min(abs(diff(data.pings(Fi).time)));
    
    % Check if the lengths of the range and time vectors are different and
    % create a unique time vector
    t_all=[];
    for ch = 1:length(F)
        % Range vector length for the different frequencies
        range(ch)=length(data.pings(ch).range);
        timerange(ch)=length(data.pings(ch).time);
        % Make a rounded time vector per frequency
        tround{ch} = round(data.pings(ch).time/tol);
        % Merge the unique time with the other frequencies
        t_all = [t_all tround{ch}];
    end
    % Create a unique timevector across frequencies
    t_final = unique(t_all);
    
    % Generate the range vector
    % Check if primary frequency target range vector can be used
    % The allowed difference is determined by the parameter par.dz and 
    % par.dzdiff
    if (median(diff(data.pings(Fi).range)) - par.dz) > par.dzdiff
        % The resolution is higher in the data than target range ->
        % average
	R = (data.pings(Fi).range(1):par.dz:data.pings(Fi).range(end))';
    elseif (median(diff(data.pings(Fi).range)) - par.dz) < -par.dzdiff
        % The resolutin is lower in the data than target range ->
        % interpolate
        R = (data.pings(Fi).range(1):par.dz:data.pings(Fi).range(end))';
    else
        % The resolution in the data for the main frequency is ok -> keep
        % range vector for Fi
        R = data.pings(Fi).range;
    end
    
    % Initialize the sv structure
    sv = zeros(length(R),length(t_final),length(F));
    % Fill in the missing pings as NaN's (in time)
    for ch = 1:length(F)
        % Keep the range but change the time vector
        sv_dum{ch}=NaN(size(data.pings(ch).sv,1),length(t_final));
        % Find the pings and add to the structure
        [~,LOCB] = ismember(tround{ch},t_final);
        % Add data to new structure
        sv_dum{ch}(:,LOCB)= data.pings(ch).sv;
    end
    % regrid the range
    for ch = 1:length(F)
        if length(R)==range(ch)
            % use the existing dta without any regridding
            fprintf('%s ','No regrid, ')
            sv(:,:,ch)=sv_dum{ch};
        elseif length(R)<range(ch)
            % the data needs to be averaged
            % Average    discretize(x,edges)
            fprintf('%s ','Averaging, ')
            edges=[R'-.5*par.dz R(end)+.5*par.dz];
            bins = discretize(data.pings(ch).range, edges);
            % If the secondary frequency has data that is outside the
            % edges it needs to be removed when averaging
            nonanid=~isnan(bins);
            for p=1:size(sv_dum{ch},2)
                sv(:,p,ch)=accumarray(bins(nonanid), sv_dum{ch}(nonanid,p), [], @mean);
            end
        elseif length(R)>range(ch)
            % The data needs to be interpolated
            fprintf('%s ','Interpolating, ')
            sv(:,:,ch)=interp1(data.pings(ch).range, data.pings(ch).sv,R);   
        end
    end
    disp(' ')
    
    %% Extract the main binary layer
    [X,Y] = meshgrid(1:size(sv,2),R);
    I = zeros(size(X));
    
    if ~isempty(snap) && ~isempty(school)
        % Loop over schools
        for i=1:length(school)
            % Plot only non empty schools (since we do not know whether an
            % empty school is assiciated to a frequency)
            if isfield(school(i),'channel')&&~isempty(school(i).channel)
                % Loop over channels
                % Plot only the relevant frequency
                
                % Get the ID string for this patch and freq
                fraction = [];% zeros(length(layer(i).school),length(length(layer(i).school(ch).species)));
                id = [];%zeros(length(layer(i).school),length(length(layer(i).school(ch).species)));
                for ch = 1:length(school(i).channel)
                    if isfield(school(i).channel(ch),'species')
                        for sp=1:length(school(i).channel(ch).species)
                            fraction(ch,sp) =str2num(school(i).channel(ch).species(sp).fraction);
                            id(ch,sp)=str2num(school(i).channel(ch).species(sp).speciesID);
                        end
                    end
                    if length(unique(id(:)))~=1
                        warning('Different IDs in layers for same layer. Using max fraction layer.')
                    end
                    % Set the species ID to the max fraction
                    [~,ind]=max(fraction(:));
                    in=inpolygon(X,Y, school(i).x,school(i).y-td);
                    if ~isempty(ind)%In some case there are no species attributed. 
                        I(in) = id(ind);
                    end
                end
            end
        end
    end
    
    % Time and reduced range vectors based on maxrange
    t=data.pings.time;
    rind=R<maxrange;
    range = R(rind);
    I = I(rind,:);
    sv = sv(rind,:,:);
    
    % Interpolate depth vector onto time vector for the data matrix
    depths = interp1(bottom.pings.time',bottom.pings.bottomdepth',t);

    % Save the data to the mat file
    save(datfile,'-v7','I','sv','F','t','range','depths','heave','trdepth')
    
    %% Write the label figure
    if plt
        fn=figure;
        hold on
        axis ij
        % Categories:
        IDstr = {'','Sandeel','Other','Possible s.e.','0-group'};
        ID  = [27 1 6009 5027];
        Ipl = I;
        Ipl(I==27)   = 1;
        Ipl(I==1)    = 2;
        Ipl(I==6009) = 3;
        Ipl(I==5027) = 4;
        % Plot
        imagesc(1:length(t),range,Ipl,[-.5 4.5])
        cmap = [1 1 1;1 0 0;0 1 0;0 0 1; 0 1 1];
        colormap(cmap)
        h=colorbar;
        set(h,'Ticks',[0 1 2 3 4],'TickLabels',IDstr)
        xlabel('ping')
        ylabel('range (m)')
        ch = find(F==(str2num(f)));
        plot(bottom.pings.bottomdepth(ch,:));
        axis tight
        print(png_I,'-dpng')
        close(fn)
        %% Write the label figure AND the raw data
        for fn=1:length(F)
            fi=figure;
            hold on
            axis ij
            % Categories:
            IDstr = {'','Sandeel','Other','Possible s.e.','0-group'};
            ID  = [27 1 6009 5027];
            Ipl = I;
            Ipl(I==27)   = 1;
            Ipl(I==1)    = 2;
            Ipl(I==6009) = 3;
            Ipl(I==5027) = 4;
            % Plot
            %imagesc(1:length(t),range,Ipl,[-.5 4.5])
            dum = 10*log10(squeeze(sv(:,:,fn)));
            dum(Ipl~=0)=max(dum(:))+10;
            imagesc(1:length(t),range,dum)
            colormap([parula;[1 0 0]])
            xlabel('time')
            ylabel('range (m)')
            plot(1:length(t),depths(:,fn),'k');
            axis tight
            print([png_I2(1:end-4),'_F',num2str(F(fn)),'kHz.png'],'-dpng')
            close(fi)
        end
    end
    
end

function ind=overlapind(I,par,sv)
%
% ind(:,1) xindex
% ind(:,2) yindex
% ind(:,3) xstep
% ind(:,4) ystep
% ind(:,5) number of nonzero classes
%
%par.dx = 400;%px
%par.dy = 400;%px
%par.overlapx = 200;%px
%par.overlapy = 200;%px
%%
S=size(I);
N1 = floor((S(1)-par.overlapx)/(par.dx-par.overlapx));
N2 = floor((S(2)-par.overlapy)/(par.dy-par.overlapy));

ind = zeros(N1*N2,5);
%% Get indices
for i=1:N1
    for j=1:N2
        ind((i-1)*N2+j,1:4) = [(par.dx-par.overlapx)*(i-1)+1,(par.dy-par.overlapy)*(j-1)+1,par.dx,par.dy];
    end
end

%% Count the number of non zero classes
for k=1:size(ind,1)
    Idum = I(ind(k,1):(ind(k,1)+ind(k,3)),ind(k,2):(ind(k,2)+ind(k,4)));
    ind(k,5) = sum(Idum(:)~=0);
end

%% Debug
% clf
% for k=1:size(ind,1)
%     if ind(k,5)>0
%         figure(1)
%         clf
%         imagesc(I(ind(k,1):(ind(k,1)+ind(k,3)),ind(k,2):(ind(k,2)+ind(k,4))))
%         figure(2)
%         clf
%         imagesc(10*log10(sv(ind(k,1):(ind(k,1)+ind(k,3)),ind(k,2):(ind(k,2)+ind(k,4)))))
%         pause(1)
%     end
% end



function finaldata=insertNaN(data,diffthreshold)
% FUNCION  insertNaN: Used to insert NaN values into a vector or matrix
% when the difference value of successive points exceed an input threshold.
%  If vector data is provided, NaN's are inserted where the differences
%  exceed the requested threshold. 
%  If matrix data is input, the difference condition is applied to the
%  first column of data, and NaN's are inserted along the entire row.
%
% usage:  output=insertNaN(data,threshold);
%   INPUTS:     data - input data which will be checked for gaps
%               threshold - threshold value to distinguish where NaN's are
%               placed in the data
%   OUTPUTS:    output - input data, with NaN values inserted where
%               differences exceeded the requested threshold
% 
% Example 1:
% output=insertNaN([11:13 15:17 19:21 25:27],1);
%      returns: 
%      output = [11 12 13 NaN 15 16 17 NaN 19 20 21 NaN 25 26 27]
% Example 2:
%  output=insertNaN([[1:2 5:7 9:10].',[1:7].',[11:17].'],1);
%      returns:
%      output =
%         1     1    11
%         2     2    12
%       NaN   NaN   NaN
%         5     3    13
%         6     4    14
%         7     5    15
%       NaN   NaN   NaN
%         9     6    16
%        10     7    17
% 
% Chris Miller
%cwmiller@nps.edu
% 9/14/11

if isvector(data), 
    diffdata=diff(data);
    index=find(diff(data)>diffthreshold);
    if isempty(index),
        finaldata=data;
        return;
    end;
    finaldata=NaN*ones(1,length(data)+length(index));  % preallocate output 
    finaldata(1:index(1))=data(1:index(1));
    if length(index)>1,
        for i=2:length(index),
            finaldata(index(i-1)+i:index(i)+i-1)=data(index(i-1)+1:index(i));
        end;
    else
        i=1;
    end;
    finaldata(index(i)+i+1:length(finaldata))=data(index(i)+1:length(data));
else,
    diffdata=diff(data(:,1));
    index=find(diffdata>diffthreshold);
    if isempty(index),
        finaldata=data;
        return;
    end;
    [n,m]=size(data);
    finaldata=NaN*ones(n+length(index),m);  % preallocate output 
    finaldata(1:index(1),:)=data(1:index(1),:);
    if length(index)>1,
        for i=2:length(index),
            finaldata(index(i-1)+i:index(i)+i-1,:)=data(index(i-1)+1:index(i),:);
        end;
    else
        i=1;
    end;
    finaldata(index(i)+i+1:length(finaldata),:)=data(index(i)+1:length(data),:);
end;
