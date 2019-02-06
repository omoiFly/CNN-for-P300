window=240; % window after stimulus (1s)

responses = zeros(12, 15, 240, 64, 100);

% convert to double precision
Signal=double(Signal);
Flashing=double(Flashing);
StimulusCode=double(StimulusCode);

% for each character epoch
for epoch=1:size(Signal,1)
    % get reponse samples at start of each Flash
    rowcolcnt=ones(1,12);
    block = 1;
    for n=2:size(Signal,2)
        if Flashing(epoch,n)==0 && Flashing(epoch,n-1)==1
            rowcol=StimulusCode(epoch,n-1);
            responses(rowcol,rowcolcnt(rowcol),:,:,epoch)=Signal(epoch,n-24:n+window-25,:);
            rowcolcnt(rowcol)=rowcolcnt(rowcol)+1;
        end
    end
end

responses = reshape(mean(responses, 2), 12, 240, 64, 100);