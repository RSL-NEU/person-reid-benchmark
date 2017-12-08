function [ExtentionArrayWithClassLabel] = StandardExtendClassLableArray(CountOfEachObjectArray)
%% <MatlabProcedure name = "" version = "1.0">
%% <About>
%%  <Author>WeiShi</Author>
%%  <Date></Date>
%%  <RevisedDate></RevisedDate>
%%  <Email>SunnyWeiShi@163.com</Email>
%%  <Work>SUN YAT-SEN UNIVERSITY</Work>
%% </About>
%% <Refer>
%%  <article></article>
%% </Refer>
%% <Function>let endpoint = startpoint + CountOfEachObjectArray(i) - 1; ExtentionArrayWithClassLabel(startpoint : endpoint) = i;  startpoint = endpoint + 1;</Function>
%% <InputParams>
%%  <param name = "ExtentionArrayWithClassLabel" type = "array">
%%    <content>see function for detail</content>
%%  </param>
%% </InputParams>
%% <OutputParams>
%%  <param name = "CountOfEachObjectArray" type = "array">
%%    <content>see function for detail</content>
%%  </param>
%% </OutputParams>
%% <Use></Use>

%% <Program>
ClassCount = length(CountOfEachObjectArray);
SampleCount = sum(CountOfEachObjectArray);
ExtentionArrayWithClassLabel = zeros(SampleCount,1);
startpoint = 1;
for i = 1 : ClassCount    
    endpoint = startpoint + CountOfEachObjectArray(i) - 1;
    ExtentionArrayWithClassLabel(startpoint : endpoint) = i;
    startpoint = endpoint + 1;
end
%% </Program>
%% </MatlabProcedure>