function specgram = generate_spectrogram(y_chunk)

    %  Size of a segment of data used to calculate each frame. This determines the
    %  basic frequency resolution of the spectrogram (128).
    segsize = 256; % N points FFT/2
    % Number of hamming windows overlapping a point (8).
    nlap = 8;
    % Factor by which transform is bigger than segment (4).
    ntrans = 4;

    [specgram, ~] = toolbox_spectrogram(y_chunk, segsize, nlap, ntrans);

end

