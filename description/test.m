
load ('data_test.mat');
test_data=[];

% responses_predict=permute(responses_test,[1,4,3,2]);
% responses_predict=reshape(responses_predict,12,100,15360);
temp=permute(responses_test,[2,1,3,4]);
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

responses_preidict=[data_temp1;data_temp2;data_temp3;data_temp4;data_temp5;data_temp6;data_temp7;data_temp8;data_temp9;data_temp10;data_temp11];


responses_preidict=permute(responses_preidict,[2,3,1]);
responses_preidict=reshape(responses_preidict,12,100,2640);


for m=1:12
    for n=1:100      
       test_data = [test_data; responses_preidict(m, n,:)];   
    end
end

test_data = reshape(test_data, 1200, 2640);

[label,score] = predict(Model_A,test_data);

