function dispfeat = featureMap(featname)

if ~iscell(featname)
    featname = {featname};
end
for f = 1:numel(featname)
    tmpname = strrep(featname{f},'lomo','LOMO');
    tmpname = strrep(tmpname,'ldfv','LDFV');
    tmpname = strrep(tmpname,'color_texture','ELF');
    tmpname = strrep(tmpname,'hist_LBP','histLBP');
    tmpname = strrep(tmpname,'sdc','DenColor');
    tmpname = strrep(tmpname,'gbicov','gBiCov');
    tmpname = strrep(tmpname,'AlexNet_Finetune','AlexNet');
    tmpname = strrep(tmpname,'gog','GOG');
    tmpname = strrep(tmpname,'IDE_ResNet', 'IDEResNet');
    tmpname = strrep(tmpname,'IDE_CaffeNet', 'IDECaffeNet');
    tmpname = strrep(tmpname,'IDE_VGGNet', 'IDEVGGNet');
    tmpname = strrep(tmpname,'whos','WHOS');
    dispfeat{f} = tmpname;
end

