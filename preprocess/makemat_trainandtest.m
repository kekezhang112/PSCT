
clear
clc

disimages = 'D:\PSCT_github\example_dataset\trainandtest\disimgs\'; %change as needed
file_pathdis = dir(fullfile(disimages));
filenamesdis = {file_pathdis.name}'; 
filenamesdis(ismember(filenamesdis,{'.','..'})) = [];
lengthdis = size(filenamesdis,1);

refimages = 'D:\PSCT_github\example_dataset\trainandtest\refimgs\'; %change as needed
file_pathref = dir(fullfile(refimages));
filenamesreftmp = {file_pathref.name}'; 
filenamesreftmp(ismember(filenamesreftmp,{'.','..'})) = [];
lengthref = size(filenamesreftmp,1);
filenamesref = cell(lengthref,1);

for l = 1:lengthdis
    for m = 1:lengthref
        TF2 = contains(filenamesdis{l}(1:end-4),filenamesreftmp{m}(1:5)); %change as needed
        if TF2 == 1
            filenamesref{l,1} = filenamesreftmp{m};
        end
    end
end


SS_all = importdata('example_MOS.txt');%change as needed
ref_ids = (1:1:lengthdis)'; 
index_all = zeros(10,lengthdis); %change as needed

for i=1:10
    index_2 = randperm(lengthdis,lengthdis);
    index_all(i,:)=index_2; 
end

save('matexample_trainandtest','SS_all','index_all','ref_ids','filenamesdis','filenamesref','-v7.3')










