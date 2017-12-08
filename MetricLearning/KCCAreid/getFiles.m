if ~exist('data/VIPeR_split.mat','file')
   disp('>Downloading and Unzipping Splits and Descriptors');
   unzip('http://www.micc.unifi.it/lisanti/downloads/kccareid_data.zip');
   disp('>Splits and Descriptors ready.')
end
if ~exist('kcca_package','dir')
   disp('>Downloading and Unzipping KCCA package');
   untar('http://www.davidroihardoon.com/Professional/Code_files/kcca_package.tar.gz');
   disp('>KCCA package ready.')
end
