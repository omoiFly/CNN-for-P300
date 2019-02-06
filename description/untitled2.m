feat = [];
for ep=1:85
    for rowcol=1:12
        ica = rica(responses(rowcol, :, 11, ep)', 3);
        feat = [feat; ica.TransformWeights];
    end
end