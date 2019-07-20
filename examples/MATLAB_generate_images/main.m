% addpath('AuditoryToolBox');
clc; clear all; close all;

%% Define global variables

image_type = 'spectrogram';

config.dir_audios = '../data/processed_audios/';
ts = datestr(now,'dd_mm_yyyy_HH_MM_SS');
dir_name = strcat(image_type, '_', ts, '/');
config.dir_images = ['../data/images/', dir_name];
config.debug = false;
config.fs = 16000;
config.win = config.fs * 1;
config.step = config.win / 2;

%% Generating images

cd ../data/images/
if ~exist(dir_name,'dir'), mkdir(dir_name), end
cd ../../generate_images

try
    generate_images(config, image_type);    
    
catch e
    fprintf('\n\n')
    disp(e)
    fprintf('Error while generating images');
end
