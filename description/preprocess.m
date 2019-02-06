window=120; % window after stimulus (0.5s)

responses = zeros(12, 15, window, 64, 85);
is_stimulate = zeros(12, 15, 85);

% convert to double precision
Signal=double(Signal);
Flashing=double(Flashing);
StimulusCode=double(StimulusCode);
StimulusType=double(StimulusType);

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
            if StimulusType(epoch, n-1) == 1
                is_stimulate(rowcol, block, epoch) = 1;               
            end
        end
        if mod(n, 504) == 0
            block = block + 1;
        end
    end
end


responses = reshape(mean(responses, 2), 12, window, 64, 85);
is_stimulate = uint8(reshape(mean(is_stimulate, 2), 12, 85));