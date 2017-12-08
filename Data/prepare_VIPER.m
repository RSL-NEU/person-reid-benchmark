clc
clear 
%%
camIDs = {'a', 'b'};
for c = 1:numel(camIDs)
    fprintf('Parsing cam-%d...\n',c);
    camID = camIDs{c};
    iminfo = dir(sprintf('./VIPeR/cam_%s/*.bmp',camID));
    names = {iminfo.name};

    for i = 1:numel(names)
        tmpname = names{i};
        mkdir(sprintf('./VIPeR/cam%d/%s',c,tmpname(1:3)));
        movefile(sprintf('./VIPeR/cam_%s/%s',camID,tmpname),sprintf('./VIPeR/cam%d/%s/%s',c,tmpname(1:3),tmpname));
    end
end