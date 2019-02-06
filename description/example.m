%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%                                                          %%%%%%
%%%%%         sample classification ofthe P300 test data       %%%%%%
%%%%%                                                          %%%%%%
%%%%%                BCI Competition III Challenge             %%%%%%
%%%%%                                                          %%%%%%
%%%%%        (C) Dean Krusienski and Gerwin Schalk 2004        %%%%%%
%%%%%                 Wadsworth Center/NYSDOH                  %%%%%%
%%%%%                                                          %%%%%%
%%%%%               function calls: topoplotEEG.m              %%%%%%
%%%%%                                                          %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all; clear all; clc

fprintf(1, '*********************************************  \n' );
fprintf(1, ' Sample classification of the P300 test data  \n' );
fprintf(1, '        BCI Competition III Challenge \n' );
fprintf(1, ' (C) Dean Krusienski and Gerwin Schalk 2004 \n' );
fprintf(1, '           Wadsworth Center/NYSDOH \n' );
fprintf(1, '*********************************************  \n\n' );

TargetChar=[];
StimulusType=[];

fprintf(1, 'Collecting Responses and Performing classification... \n\n' );
load 'Subject_A_Train.mat' % load data file
window=240; % window after stimulus (1s)
channel=11; % only using Cz for analysis and plots

% convert to double precision
Signal=double(Signal);
Flashing=double(Flashing);
StimulusCode=double(StimulusCode);
StimulusType=double(StimulusType);

% 6 X 6 onscreen matrix
screen=char('A','B','C','D','E','F',...
            'G','H','I','J','K','L',...
            'M','N','O','P','Q','R',...
            'S','T','U','V','W','X',...
            'Y','Z','1','2','3','4',...
            '5','6','7','8','9','_');

% for each character epoch
for epoch=1:size(Signal,1)
    
    % get reponse samples at start of each Flash
    rowcolcnt=ones(1,12);
    for n=2:size(Signal,2)
        if Flashing(epoch,n)==0 && Flashing(epoch,n-1)==1
            rowcol=StimulusCode(epoch,n-1);
            responses(rowcol,rowcolcnt(rowcol),:,:)=Signal(epoch,n-24:n+window-25,:);
            rowcolcnt(rowcol)=rowcolcnt(rowcol)+1;
        end
    end

    % average and group responses by letter
    m=1;
    avgresp=mean(responses,2);
    avgresp=reshape(avgresp,12,window,64);
    for row=7:12
        for col=1:6
            % row-column intersection
            letter(m,:,:)=(avgresp(row,:,:)+avgresp(col,:,:))/2;
            % the crude avg peak classifier score (**tuned for Subject_A**)          
            score(m)=mean(letter(m,54:124,channel))-mean(letter(m,134:174,channel));
            m=m+1;
        end
    end
    
    [val,index]=max(score);
    charvect(epoch)=screen(index);
    
    % if labeled, get target label and response
    if isempty(StimulusType)==0
        label=unique(StimulusCode(epoch,:).*StimulusType(epoch,:));
        targetlabel=(6*(label(3)-7))+label(2);
        Target(epoch,:,:)=.5*(avgresp(label(2),:,:)+avgresp(label(3),:,:));
        NonTarget(epoch,:,:)=mean(avgresp,1)-(1/6)*Target(epoch,:,:);
    end
end

% display results

if isempty(TargetChar)==0

    k=0;
    for p=1:size(Signal,1)
        if charvect(p)==TargetChar(p)
            k=k+1;
        end
    end

    correct=(k/size(Signal,1))*100;

    fprintf(1, 'Classification Results: \n\n' );
    for kk=1:size(Signal,1)
        fprintf(1, 'Epoch: %d  Predicted: %c Target: %c\n',kk,charvect(kk),TargetChar(kk));
    end
    fprintf(1, '\n %% Correct from Labeled Data: %2.2f%% \n',correct);

    % plot averaged responses and topography
    Tavg=reshape(mean(Target(:,:,:),1),window,64);
    NTavg=reshape(mean(NonTarget(:,:,:),1),window,64);
    figure
    plot([1:window]/window,Tavg(:,channel),'linewidth',2)
    hold on
    plot([1:window]/window,NTavg(:,channel),'r','linewidth',2)
    title('Averaged P300 Responses over Cz')
    legend('Targets','NonTargets');
    xlabel('time (s) after stimulus')
    ylabel('amplitude (uV)')
    
    % Target/NonTarget voltage topography plot at 300ms (sample 72)
    vdiff=abs(Tavg(72,:)-NTavg(72,:));
    figure
    topoplotEEG(vdiff,'eloc64.txt','gridscale',150)
    title('Target/NonTarget Voltage Difference Topography at 300ms')
    caxis([min(vdiff) max(vdiff)])
    colorbar
    
else

    for kk=1:size(Signal,1)
        fprintf(1, 'Epoch: %d  Predicted: %c\n',kk,charvect(kk));
    end

end

fprintf(1, '\nThe resulting classified character vector is the variable named "charvect". \n');
fprintf(1, 'This is an example of how the results *must* be formatted for submission. \n');
fprintf(1, 'The character vectors from each case and subject are to be labeled, grouped, and submitted according to the accompanied documentation. \n');
