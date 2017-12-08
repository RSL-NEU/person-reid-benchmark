function p=haveDone()

k=dir('../Results/');
k=k(3:end);
p = {''};
for i=1:numel(k)
    currstr=k(i).name;
    ind=findstr(currstr,'2017');
    currstr=currstr(1:ind-1);
    p{i}=currstr;
end

tmp=1;