function [idx_cam_a,idx_cam_b] = partition(fileread_info)

%% Partition

Img_idx = zeros(length(fileread_info),1);
Img_id = zeros(length(fileread_info),1);

temp_idx = regexp(fileread_info,'cam_a','start');

expr = '/(\d{3})';
temp_id = regexp(fileread_info,expr,'tokens');

for i = 1: length(fileread_info)
    Img_idx(i) = not(isempty(temp_idx{i}));
    Img_id(i) = str2num(temp_id{i,1}{1,1}{1,1});
%     Img_pos(i) = str2num(temp_pos{i,1}{1,1}{1,1});
end


%% alignment

idx_cam_a = find(Img_idx == 1);
idx_cam_b = find(Img_idx == 0);

Img_id_a = Img_id(idx_cam_a);
Img_id_b = Img_id(idx_cam_b);
[Img_id_a,IX_a]=sort(Img_id_a);
[Img_id_b,IX_b]=sort(Img_id_b);
idx_cam_a = idx_cam_a(IX_a);
idx_cam_b = idx_cam_b(IX_b);

if ~isequal(Img_id_a, Img_id_b) %(sum(Img_id_a == Img_id_b) ~= length_cam_a)
% if ~isequal(idx_cam_a,idx_cam_b)
   error('cam_a and cam_b index is not aligned\n');
end


end