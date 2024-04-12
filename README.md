# chineseTonesDetector

## Download files

### setup environment
```sh
conda create --name chineseTones_env python=3.8
conda activate chineseTones_env
pip install requests
pip install jupyter 
pip install numpy matplotlib librosa pandas seaborn tensorflow
```

### download data samples
```sh
cd prepareData
python downloadTonesData.py
```

### analyse data samples
```sh
cd prepareData
analyzeTones.ipynb
```

## ML API

### setup environment
```sh
pip install librosa Flask flask-cors pydub
(brew install ffmpeg)
```

### run
```sh
cd flaskAPI
python spectrum.py
```

## Website

### setup environment
```sh
npx create-react-app website
npm install recordrtc react-audio-player react-router-dom
npm install @mui/material @emotion/react @emotion/styled
```

### run
```sh
cd frontend
npm start
```
