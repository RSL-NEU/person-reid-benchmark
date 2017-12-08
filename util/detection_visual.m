function im_exp = detection_visual(I,num_col,max_disp,perimsz)
% im_exp = detection_visual(I,num_col,max_disp,perimsz)
% 
if numel(I) > max_disp
    I = I(1:max_disp);
end    
% num_col = 30;
% subtestID = testID(subQxx(1:subr));
% displayID = [showID(s) subtestID];
max_row = ceil(numel(numel(I))/num_col);
im_exp = zeros(perimsz(1)*max_row,perimsz(2)*num_col,3);
for id = 1:numel(I)
    tmp_row = ceil(id/num_col);
    tmp_col = mod(id,num_col);
    if tmp_col==0
        tmp_col=num_col;
    end
    if iscell(I{id})
        imsz = cellfun(@size,I{id},'UniformOutput',0);
        imsz = cell2mat(imsz');
        imsz = imsz(:,1:2);
        imsz = imsz(:,1).*imsz(:,2);
        tmpI = I{id}{imsz==max(imsz)};
    else 
        tmpI = I{id};
    end
    tmpI = imresize(tmpI,perimsz);
        
%     if id == 1
%         tmpI = seq_prob;
%     else
%         tmpI = I{gID==displayID(id)&camID==camg};
%     end            
%     tmpI = tmpI{round(numel(tmpI)/2)};
    im_exp((tmp_row-1)*perimsz(1)+1:tmp_row*perimsz(1),(tmp_col-1)*perimsz(2)+1:tmp_col*perimsz(2),:) = tmpI;
end  
im_exp = uint8(im_exp);