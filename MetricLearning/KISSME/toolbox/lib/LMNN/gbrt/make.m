% mex buildlayer_sqrimpurity_openmp.cpp CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"
% mex buildlayer_sqrimpurity_openmp_multi.cpp CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"
% mex buildlayer_sqrimpurity_openmp_multi_limit.cpp CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"

mex buildlayer_sqrimpurity_openmp.cpp 
mex buildlayer_sqrimpurity_openmp_multi.cpp 
mex buildlayer_sqrimpurity_openmp_multi_limit.cpp 