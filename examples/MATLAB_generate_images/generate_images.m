function generate_images(config, image_type)

    listAudioFiles = dir([config.dir_audios, '*.wav']);
    nAudios = length(listAudioFiles);    
    
    %%%%%%%%%%%%%%%%%%
    % For each audio %
    %%%%%%%%%%%%%%%%%%
    for n=1:nAudios
        fprintf("\n============================");
        fprintf("\nAudio %d / %d", n, nAudios);
        filename = [config.dir_audios, listAudioFiles(n).name];
        [y, ~] = audioread(filename);
        totalLength = length(y);
        
        % With overlapping
%         nChunks = ceil(totalLength / config.step);

        % Without overlapping
        nChunks = ceil(totalLength / config.win);
        
        fprintf('  -  %s - %d samples - %d chunks', listAudioFiles(n).name, totalLength, nChunks);

        %%%%%%%%%%%%%%%%%%
        % For each chunk %
        %%%%%%%%%%%%%%%%%%
        for k=1:nChunks
            fprintf("\n\t * Chunk %d / %d", k, nChunks);
            
            % With overlapping
%             i_start = (k-1)*config.step + 1;

            % Without overlapping
            i_start = (k-1)*config.win + 1;
            
            if (i_start + config.win) < totalLength
                i_end = i_start + config.win - 1;
            else
                i_end = totalLength;
            end
            
            fprintf(" - From %d to %d", i_start, i_end);
            y_chunk = y(i_start:i_end);
            
            %%%%%%%%%%%%%%%%%%
            % Generate image %
            %%%%%%%%%%%%%%%%%%
            switch image_type
                case "spectrogram"
                    im = generate_spectrogram(y_chunk);  
                    [xmax, ymax] = size(im);
                    
                    yLabel = 'Frequency (Hz)';
                    xLabel = 'Time (samples)';                    
                    yRange = linspace(config.fs/2, 0, ymax);                    
                    xRange = linspace(0, length(y_chunk), xmax);
                otherwise
                    fprintf("\nNot implemented yet.");
            end
            
            %%%%%%%%%%%%%%
            % Show image %
            %%%%%%%%%%%%%%
            figure();                
            imagesc(xRange, yRange, im);
            set(gca,'YDir','normal')
            xlabel(xLabel);
            ylabel(yLabel);
            title(image_type);
            
            %%%%%%%%%%%%%%
            % Save image %
            %%%%%%%%%%%%%%
            name = split(listAudioFiles(n).name, '.');
            filesave = [config.dir_images, name{1}, '_', num2str(k), '.png'];
            
%             set(gca,'XTick',[]) % Remove the ticks in the x axis
%             set(gca,'YTick',[]) % Remove the ticks in the y axis
%             set(gca,'Position',[0 0 1 1]) % Make the axes occupy the hole figure
%             saveas(gcf,filesave,'jpg')
%             close all;
            
            imwrite(ind2rgb(im2uint8(mat2gray(im)), parula(512)), filesave)
                         
            
        end % end for k=1:nChunks
        
    end % end for n=1:nAudios
    
end % end function generate_images

