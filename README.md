# chineseTonesDetector

## Download files

### setup environment
```sh
conda create --name chineseTones_env python=3.8
conda activate chineseTones_env
pip install jupyter 
pip install requests numpy matplotlib librosa pandas seaborn tensorflow boto3
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


## AWS

###
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_DEFAULT_REGION="your_region"

