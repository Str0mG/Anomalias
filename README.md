<h4 align="center"> 
	🚧  Algoritmo de movimento 🚀 Em construção...  🚧
</h4>

# Analise de algoritmo de detecção de movimento

Nesse projeto vamos construir uma rede neural que analisa se há uma anomalia ou não em um frame.


### 📋 Pré-requisitos

Ambiente Virtual: 
```
python3 -m venv venv
```
Entrar no ambiente virtual:
```
source venv/bin/activate
```

Requirements:
```
pip install -r requirements.txt
```

### 🔧 Quick Start

```
|-- **annotation
|   |-- files                  // Insert here your .csv files

|-- videos
|   |-- fights
|   |   |-- 0.mp4                 // Fights Videos
|   |   |-- 1.mp4
|   |   |-- ...
|   |-- normal
|   |   |-- 0.mp4                  // Normal Videos
|   |   |-- 1.mp4
|   |   |-- ...

`-- frames         
|   |-- fights
|   |   |-- 0.mp4                 // Fights Videos
|   |	|   |-- 0.png
|   |	|   |-- 1.png
|   |   |-- 1.png
|   |	|   |-- 0.png
|   |	|   |-- 1.png

|   |-- normal
|   |   |-- 0.mp4                 // Normal frames
|   |	|   |-- 0.png
|   |	|   |-- 1.png
|   |   |-- 1.png
|   |	|   |-- 0.png
|   |	|   |-- 1.png
    
```

Converter video to frames:
```
python3 convert.py --root_videos "path_to_videos"  --root_frames "path_frames"
```

Pré processamento (Extração do grayscale -> variação dos histogramas -> treinamento da rede neural):

```
python3 index.py --root_frames "path_frames" --root_csv "path_to_csv"
```


## 🛠️ Construído com

* [TensorFlow](https://www.tensorflow.org/)
* [OpenCv](https://opencv.org/)

---
⌨️ com ❤️ por [Alan Araujo - Gabriel Trombini - Pedro Lemes ]😊
