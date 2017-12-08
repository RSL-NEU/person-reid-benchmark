% [ix_pos_pair, ix_neg_pair]=GeneratePair(gID,CamID,npratio);
% given the ID set, find the positive and negative pair of samples.
% NOTE that the negative pair samples are random permutated, so that taking
% first N pairs is the same as randomly pick the negative pairs.

% - add cam ID as input to make sure the images within each pair are from
%   different cameras 
% Modified by Mengran Gou @ 08/10/2015
function [ix_pos_pair, ix_neg_pair]=GeneratePair(varargin)
ID = varargin{1};
npratio = 0;
if nargin > 1
    camID = varargin{2};
end
if nargin > 2
    npratio = varargin{3}; % negative to positive pair ratio
end
if ~exist('camID','var')
    ID = varargin{1};
    ID = double(ID);
    R = repmat(ID.^2, length(ID), 1);
    R = R + R' -2*ID'*ID;
    idx_triu = find(triu(ones(size(R)) - eye(size(R)))>0);
    idx_pos = idx_triu(R(idx_triu(:)) ==0);
    idx_neg = idx_triu(R(idx_triu(:)) ~=0);
    rndp = randperm(length(idx_neg));
    idx_neg = idx_neg(rndp);
    [ix_neg_pair(:,1), ix_neg_pair(:,2) ]= ind2sub(size(R), idx_neg);
    [ix_pos_pair(:,1), ix_pos_pair(:,2) ]= ind2sub(size(R), idx_pos);
else
    ID = varargin{1};
    camID = varargin{2};
    uniCam = unique(camID);
    ix_pos_pair = [];
    ix_neg_pair = [];
    % loop over all camera pairs
    for i = 1:numel(uniCam)-1
        for j = i+1:numel(uniCam)
            id_cam1 = find(camID==uniCam(i));
            id_cam2 = find(camID==uniCam(j));
            ID1 = ID(id_cam1);
            ID2 = ID(id_cam2);
            R1 = repmat(ID1',1,length(ID2));
            R2 = repmat(ID2,length(ID1),1);
            R = R1-R2;
            [tmp_pos1, tmp_pos2] = find(R==0);
            [tmp_neg1, tmp_neg2] = find(R~=0);
            if isempty(tmp_pos1) % invalid camera pair
                continue;
            end
            % map back to original index
            ix_pos_pair = [ix_pos_pair; cat(2,id_cam1(tmp_pos1)',id_cam2(tmp_pos2)')];
            ix_neg_pair = [ix_neg_pair; cat(2,id_cam1(tmp_neg1)',id_cam2(tmp_neg2)')];
        end
    end
end
if npratio > 0
    num_neg_pair = min(npratio*size(ix_pos_pair,1), size(ix_neg_pair,1));
    ix_neg_pair =ix_neg_pair(randsample(size(ix_neg_pair,1),num_neg_pair), :);
end
ix_neg_pair = ix_neg_pair(randperm(size(ix_neg_pair,1)),:);
return;