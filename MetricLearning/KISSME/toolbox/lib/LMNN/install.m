function install(command)
%	function install(command)
%
% call "install" to compile functions on demand (when required)
% or "install('force')"	to compile all functions (even when binaries already exist)
% 
% copyright by Kilian Q. Weinberger, Sept. 2012

if nargin<1,command='';end;
force=strcmp(command,'force');
force=false;
if nargin==1, 
	if strcmp(command,'force'), force=true;
	else 
		fprintf('Command %s unknown.\n',command);
		return;
	end;
end;

setpaths;
cd mexfunctions
fprintf('Compiling all mex functions ... \n\n');

comperror=false;
% compile helperfunctions
comperror=comperror || compile('count.c',force);
comperror=comperror || compile('sd.c',force);
comperror=comperror || compile('sd2.c',force);
comperror=comperror || compile('sd2b.c',force);
comperror=comperror || compile('SODd.c',force);
comperror=comperror || compile('sumiflessh2.c',force);
comperror=comperror || compile('sumiflessv2.c',force);
comperror=comperror || compile('mink.c',force);
comperror=comperror || compile('minkomp.c',force);
comperror=comperror || compile('cdist.c',force);
comperror=comperror || compile('cdistomp.c',force);
comperror=comperror || compile('SODW.c',force);
comperror=comperror || compile('SODmex.c',force);
comperror=comperror || compile('lmnnobj.cpp',force);

cd ..

% compile mtrees
cd mtrees
comperror=comperror || compile('buildmtreec.cpp',force);
comperror=comperror || compile('findknnmtree.cpp',force);
comperror=comperror || compile('findknnmtreeomp.cpp',force);
comperror=comperror || compile('findNimtree.cpp',force);
cd ..

cd gbrt
comperror=comperror || compile('buildlayer_sqrimpurity.cpp',force);
comperror=comperror || compile('buildlayer_sqrimpurity_multif.cpp',force);
comperror=comperror || compile('buildlayer_sqrimpurity_openmp.cpp',force);
comperror=comperror || compile('buildlayer_sqrimpurity_openmp_multi.cpp',force);
comperror=comperror || compile('buildlayer_sqrimpurity_openmp_multi_limit.cpp',force);
cd ..

if 	comperror,
	fprintf('\n\nLooks like you had some compilation errors.\n');
	fprintf('There is a good chance it still works with the pre-compiled binaries that are already included in the path.\nGood luck!\n')
else
	fprintf('\n\nNo compilation errors. Everything should run smoothly.\n')
	if ~force,
 	 fprintf('(In case of errors you can force recompilation with "install(''force'')".)\n')
   end;
   fprintf('Run "demo" to try out LMNN.\nGood luck!\n-Kilian\n');
end;



function comperror=compile(filename,force)
	
	comperror=false;
	s=regexp(filename,'\.','split');
	fprintf('%s...',filename)
	if ~force && (exist(['./' s{1}])==3 || exist(['binaries/' s{1}])==3),
		fprintf('[binary exists]\n');
		return;
	end;
	try
	  eval(['mex ' filename ' ' 'CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"']);
		fprintf('[compiled]\n');
	catch
		fprintf('[ERROR]\n');
        fprintf(lasterr);
        fprintf('\n');
		comperror=true;
	end;    
	
