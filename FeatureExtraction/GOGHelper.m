function x=GOGHelper(I,param)

x=[];
for i=1:numel(I)
    if mod(i,round(numel(I)/10))
        fprintf('.');
    end  
    x(i,:)=GOG(I{i},param);
end