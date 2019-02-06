figure
for ep=1:85
    for rowcol=1:12
       idx = rowcol*ep;
       if is_stimulate(rowcol, ep) == 1
           scatter3(feat(idx,1), feat(idx,2), feat(idx,3), 'r')
           hold on
       else
           scatter3(feat(idx,1), feat(idx,2), feat(idx,3), 'b')
           hold on
       end
    end
end