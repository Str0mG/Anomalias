import os, re
import argparse
import cv2
import numpy as np
import pandas as pd
from scipy.spatial import distance


def humanSort(text):  # Sort function for strings w/ numbers
    convText = lambda seq: int(seq) if seq.isdigit() else seq.lower()
    arrayKey = lambda key: [convText(s) for s in re.split('([0-9]+)', key)]  # Split numbers and chars, base function for sorted
    return sorted(text, key=arrayKey)


def distE(frames:list):
    dist = []
    for i in range(len(frames)-1):
        dist.append(distance.euclidean(frames[i], frames[i+1]))
    return dist
    

def rgb(image:list)->list:
    rgb = []

    for i in range(len(image)):
        image = cv2.imread(image[i])
        gray = cv2.calcHist([image], [0], None, [256], [0, 256])
        rgb.append(np.append([],gray).tolist())
        
    return rgb


def read_file(source):
    result = []
    with open(source, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            result.append(row)
    return result

# Verificar com mais frames para ver
def normalize(root_videos):
    frame_delta = []
    # for entre fight e normal
    for frame_dir in humanSort(os.listdir(root_videos)):
        path_dir_full = os.path.join(root_videos,frame_dir)
        # for entre videos
        for frame_file in humanSort(os.listdir(path_dir_full)):
            path_frame_full = os.path.join(path_dir_full,frame_file)
            frames_name = humanSort(os.listdir(path_frame_full))
            print(len(frames_name))
            frame_rgb = rgb(frames_name) #lista de lista para vetores de cada frame [[1,2,3,4],[1,2,4,5]...]
            frame_delta.append(distE(frame_rgb)) #lista de listas [[],[]]
    
    return frame_delta
    

def csv(root_csv):
    frame_csv = []

    for csv_file in humanSort(os.listdir(root_csv)):
        i+=1
        print(humanSort(os.listdir(root_csv)))
        path_csv_full = os.path.join(root_csv,csv_file)
        framesAnotados = read_file(path_csv_full)
        print(len(framesAnotados))

    frame_csv 


def rede_neural(dif,csv):
    # transforma para csv
    df = pd.DataFrame(dif,columns=['Diferenca'])

    df.to_csv('Eae2.csv',index=False)

if __name__ == '__main__':
    # Como rodar
    # python3 index.py --root_frames "path_frames"

    class TestFailed(Exception):
        pass
    
    parser = argparse.ArgumentParser()

    # Paths and settings
    parser.add_argument('--root_frames',    type=str, default='PATH_TO_FRAMES', help='Root path of your frames')

    parser.add_argument('--root_csv',    type=str, default='PATH_TO_CSV', help='Root path of your csv')
    opt = parser.parse_args()
    
    dif = normalize(opt.root_videos) #[[123,123],[1231,1231]]
    csv_list = csv(opt.root_csv)    #[[0,1],[0,1]]
    # funcao so para checar se ta ok
    if not len(dif) == len(csv_list):
        raise TestFailed('N esta igual')

    for i in range(len(dif)):
        if not len(dif[i]) == len(csv_list[i]):
            raise TestFailed('N esta igual')

    rede_neural(dif,csv_list)

# To-do verificar rgb de um grayscale e fazer a media e verificar se é o msm se fosse só grayscale


# Problemas como armazenar para treinar uma gama muito grande -> dar exemplos