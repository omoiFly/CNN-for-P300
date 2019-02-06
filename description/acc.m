

dis = ['A' 'B' 'C' 'D' 'E' 'F';
         'G' 'H' 'I' 'J' 'K' 'L';
         'M' 'N' 'O' 'P' 'Q' 'R';
         'S' 'T' 'U' 'V' 'W' 'X';
         'Y' 'Z' '1' '2' '3' '4';
         '5' '6' '7' '8' '9' '_'];

target=('WQXPLZCOMRKO97YFZDEZ1DPI9NNVGRQDJCUVRMEUOOOJD2UFYPOO6J7LDGYEGOA5VHNEHBTXOO1TDOILUEE5BFAEEXAW_K4R3MRU');
% load ('Predict_A.mat', 'label');
row_col=zeros(100,2);


for i=1:100
    for col=1:6
        j=col+12*(i-1);
        if label(j)==1
            row_col(i,2)=col;
            
        elseif col==6&&row_col(i,2)==0
            row_col(i,2)=1;
        end
    end  
    for row=7:12
        k=row+12*(i-1);
        if label(k)==1
            row_col(i,1)=row-6;
           
        elseif row==12&&row_col(i,1)==0
            row_col(i,1)=1;
        end
    end
    
end

is_target=[];
for i=1:100
    
    is_target(i)=dis(row_col(i,1),row_col(i,2));
    predict_char=char(is_target());
   
end
count=0;
for i=1:100
    if predict_char(i)==target(i)
    count=count+1;
    end
end    
accuracy=count/100;      
disp(sprintf('accuracy= %8.2\n',accuracy));
