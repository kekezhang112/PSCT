
clear
clc

disimages = 'D:\PSCT_github\example_dataset\onlyfortest\disimgs\'; %change as needed
file_pathdis = dir(fullfile(disimages));
filenamesdis = {file_pathdis.name}'; 
filenamesdis(ismember(filenamesdis,{'.','..'})) = [];
lengthdis = size(filenamesdis,1); 


refimages = 'D:\PSCT_github\example_dataset\onlyfortest\refimgs\'; %change as needed
file_pathref = dir(fullfile(refimages));
filenamesreftmp = {file_pathref.name}'; 
filenamesreftmp(ismember(filenamesreftmp,{'.','..'})) = [];
lengthref = size(filenamesreftmp,1); 
filenamesref = cell(lengthref,1);

for l = 1:lengthdis
    for m = 1:lengthref
        TF = contains(filenamesdis{l}(1:end-4),filenamesreftmp{m}(1:5)); %change as needed
        if TF == 1
            filenamesref{l,1} = filenamesreftmp{m};
        end
    end
end

SS_all = importdata('MOS_onlyfortest.txt'); %change as needed
ref_ids = (1:1:lengthdis)'; 
index1 = (1:1:lengthdis); 

save('matexample_onlyfortest','SS_all','index1','ref_ids','filenamesdis','filenamesref','-v7.3')











