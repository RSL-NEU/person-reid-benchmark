function progressbar(n,N,s2);
% function progressbar(n,N,s2);
%
% displays the progress of n out of N
%
% s2 is an optional string input that will get displayed after the progress bar. 
% copyright Kilian Q. Weinberger

LE=10;
persistent oldp;
if(isempty(oldp)) oldp=11;end;
p=floor(n/N*LE);
if(p~=oldp)
 if(~exist('s2')) s2='';end;
 ps='..........';
 ps(1:p)='*';
 fprintf('\rProgress:[%s]%s',ps,s2);
 oldp=p;
end;