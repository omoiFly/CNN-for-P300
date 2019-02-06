clear all;  
clc;
temp=zeros(12,85,64,240);
responses_train=zeros(12,85,11,240);
load ('data.mat');
data = [];
label = [];

temp=permute(responses,[2,1,3,4]);
data_temp1=squeeze(temp(:,:,3,:));
data_temp2=squeeze(temp(:,:,5,:));
data_temp3=squeeze(temp(:,:,9,:));
data_temp4=squeeze(temp(:,:,11,:));
data_temp5=squeeze(temp(:,:,13,:));
data_temp6=squeeze(temp(:,:,22,:));
data_temp7=squeeze(temp(:,:,24,:));
data_temp8=squeeze(temp(:,:,34,:));
data_temp9=squeeze(temp(:,:,51,:));
data_temp10=squeeze(temp(:,:,56,:));
data_temp11=squeeze(temp(:,:,60,:));

responses_train=[data_temp1;data_temp2;data_temp3;data_temp4;data_temp5;data_temp6;data_temp7;data_temp8;data_temp9;data_temp10;data_temp11];


responses_train=permute(responses_train,[2,3,1]);
responses_train=reshape(responses_train,12,85,2640);
for i=1:12
    for j=1:85    
        data = [data; responses_train(i, j,:)];
        label = [label; is_stimulate(i, j)];
    end
end

data = reshape(data, 1020, 2640);



Model_A= fitcsvm(data,label);