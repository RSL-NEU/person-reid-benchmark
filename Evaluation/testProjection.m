function [ testFeatProj ] = testProjection( testFeat, trainFeat, metric, mopts )
% Project test feature based on the metric learning method
% INPUT
%   testFeat    - [NxD]
%   trainFeat   - [MxD]
%   metric      - [struct] learned metric
%   mopts       - [struct] metric learning options
% OUTPUT
%   testFeatProj- [Nxd] prjected test feature
% default: testFeatProj = testFeat*metric.T;
% Write by Mengran Gou @ 2017

% kernel mapping
if ~isempty(mopts.kernels) && ~strcmp(mopts.method,'kcca')
    K.kernel = metric.options.kernel;
    K.rbf_sigma = metric.options.rbf_sigma;
    if 0%gpuDeviceCount > 0 && any(size(testFeat)>1e4) % use GPU for really large matrix
        testFeat = single(testFeat);
        trainFeat = single(trainFeat);        
        try
            disp('Using GPU...');
            testFeat = gpuArray(testFeat);
            [Ktest] = ComputeKernelTest(trainFeat, testFeat, K);
            Ktest = gather(Ktest);
            reset(gpuDevice()); % reset GPU memory if used
        catch
            info = whos('testFeat');
            if strfind(info.class,'gpu')
                testFeat = gather(testFeat);
            end
            reset(gpuDevice());
            disp('Compute on GPU failed, using CPU now...');
            [Ktest] = ComputeKernelTest(trainFeat, testFeat, K);
        end
    else
        [Ktest] = ComputeKernelTest(trainFeat, testFeat, K);
    end
    if ~strcmp(mopts.name,'kcca')
        testFeatProj = (metric.T*Ktest)';
    end
elseif  strcmp(mopts.method,'svmml') || ...
        strcmp(mopts.method,'kissme') || ...
        strcmp(mopts.method,'ranksvm') || ...
        strcmp(mopts.method,'xqda') || ...
        strcmp(mopts.method,'prdc') || ...
        strcmp(mopts.method,'kcca') ||...
        strcmp(mopts.method,'l2')
    testFeatProj = testFeat;
elseif strcmp(mopts.method,'sssvm')
    testFeatProj = testFeat*metric.T.P';
else
    testFeatProj = testFeat*metric.T;
end

end

