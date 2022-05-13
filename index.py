import os, re, csv
import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt
from statistics import mean
from math import sqrt

def humanSort(text):  # Sort function for strings w/ numbers
    convText = lambda seq: int(seq) if seq.isdigit() else seq.lower()
    arrayKey = lambda key: [convText(s) for s in re.split('([0-9]+)', key)]  # Split numbers and chars, base function for sorted
    return sorted(text, key=arrayKey)


def read_file(source):
    result = []
    with open(source, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            result.append(row)
    return result


def write_file(source, content):
    with open(source, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(content)


def distE(a,b):
    return sqrt(sum((e1-e2)**2 for e1, e2 in zip(a,b)))
    

def rgb(image):
    image = cv2.imread(image)
    r = cv2.calcHist([image], [2], None, [256], [0, 256])
    return r


def normalize(root_csv, root_videos):
    i=0

    #Funcção para colocar cada linha do csv  numa lista
    for csv_file in humanSort(os.listdir(root_csv)):
        i+=1
        print(humanSort(os.listdir(root_csv)))
        path_csv_full = os.path.join(root_csv,csv_file)
        framesAnotados = read_file(path_csv_full)
        print(len(framesAnotados))
    
    #Funcção para colocar frames numa lista
    framesImagem = humanSort(os.listdir(root_videos))

    print(len(framesImagem))

    frameTrue = []
    frameFalse = []
    for i in range(len(framesAnotados)):
        
        if '1' in framesAnotados[i]:
            
            r= rgb(f'/home/trombini/anomalias/projeto1/fight/F_1_0_1_0_1.mp4/{framesImagem[i]}')
            aux = []
            aux = np.append(aux,r)
            frameTrue.append(aux.tolist()) #lista de lista para vetores de cada frame [[],[]...]
        
        if '0' in framesAnotados[i]:
            r= rgb(f'/home/trombini/anomalias/projeto1/fight/F_1_0_1_0_1.mp4/{framesImagem[i]}')
            aux = []
            aux = np.append(aux,r)
            frameFalse.append(aux.tolist()) #lista de lista para vetores de cada frame [[],[]...]
    
    #Calcular o vetor médio do R para lista de lista(Não precisa)
    mediaTrue = [mean(values) for values in zip(*frameTrue)]
    mediaFalse = [mean(values) for values in zip(*frameFalse)]
    dist = np.linalg.norm(mediaTrue-mediaFalse)
    #return mediaTrue,mediaFalse,dist

    #Calcular a diferença entre cada frame (Detecção de movimento) len = len(frames - 1)
    delta = []
    for i in range(len(frameTrue)-1):
        delta.append(distE(frameTrue[i+1],frameTrue[i]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Paths and settings
    parser.add_argument('--root_csv',    type=str, default='PATH_TO_CSVs',     help='Root path of your csvs')
    parser.add_argument('--root_videos',    type=str, default='PATH_TO_VIDEOS', help='Root path of your videos')
    
    opt = parser.parse_args()

    num_frames = opt.fps * opt.duration

    a,b,c =normalize('/home/trombini/anomalias/projeto1/annotation','/home/trombini/anomalias/projeto1/fight/F_1_0_1_0_1.mp4')
    print(a,b,c)



