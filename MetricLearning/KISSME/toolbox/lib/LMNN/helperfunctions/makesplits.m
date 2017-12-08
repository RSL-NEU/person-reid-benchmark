function [train,test]=makesplits(y,split,splits,classsplit,k,dummy)
%  function [train,test]=makesplits(y,split,splits,classsplit)
%
% SPLITS "y" into "splits" sets with a "split" ratio.
% if classsplit==1 then it takes a "split" fraction from each class
%

if(split==1)
    train=randperm(length(y));
    test=[];
    return;
end;    


if(split==0)
    test=randperm(length(y));
    train=[];
    return;
end;    


if(nargin<4)
    classplit=0;
end;
if(nargin<5)
    k=1;
end;

n=length(y);
if(minclass(y,1:length(y))<k || split*length(y)/length(unique(y))<k)    
    fprintf('K:%i split:%f n:%i\n',k,split,length(y));
    keyboard;
    error('Cannot sub-sample splits! Reduce number of neighbors.');
end;


   if(classsplit)
        un=unique(y);
        for i=1:splits
            trsplit=[];
            tesplit=[];
            while(minclass(y,trsplit)<k)
            for j=1:length(un)
                ii=find(y==un(j));
                co=round(split*length(ii));
                ii=ii(randperm(length(ii)));
                trsplit=[trsplit ii(1:co)];
                tesplit=[tesplit ii(co+1:end)];
            end;        
            end;
            train(i,:)=trsplit;
            test(i,:)=tesplit;            
        end;
    else        
        for i=1:splits
            trsplit=[];
            tesplit=[];
            while(minclass(y,trsplit)<k)                
             ii=randperm(n);            
             co=round(split*n);
             trsplit=ii(1:co);
             tesplit=ii(co+1:n);
            end;
            train(i,:)=trsplit;
            test(i,:)=tesplit;            
        end;
    end;    
    
    
    

function [m]=minclass(y,ind)
% function m=minclass(y,ind)
%
%

un=unique(y);
m=inf;
for i=1:length(un)
    m=min(sum(y(ind)==un(i)),m);
end;

