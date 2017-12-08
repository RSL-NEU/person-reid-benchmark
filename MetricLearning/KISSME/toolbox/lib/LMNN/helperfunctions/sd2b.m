function [plus1,minus1,act1,plus2,minus2,act2]=sd2b(g0,g1,a1,g2,a2);
% function [plus1,minus1,act1,plus2,minus2,act2]=sd2b(g0,g1,a1,g2,a2);
%
%
% copyright Kilian Q. Weinberger, 2015

[plus1,minus1,act1]=sd2(g0,g1,a1);
[plus2,minus2,act2]=sd2(g0,g2,a2);   
