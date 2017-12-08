function p=haveDone1(currstr,estr)

p=0;
for i=1:numel(estr)
    tmp=estr{i};
    if(strcmp(currstr,tmp))
        p=1;
        break;
    end
end
