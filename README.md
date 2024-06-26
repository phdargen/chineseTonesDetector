# chineseTonesDetector

## Download files

### setup environment
```sh
conda create --name chineseTones_env python=3.8
conda activate chineseTones_env
pip install jupyter 
pip install requests numpy matplotlib librosa pandas seaborn tensorflow boto3
pip install gTTS
pip install soundfile
pip install tensorflow-macos
pip install tensorflow-metal
```

### Fine tune base ML model
```sh
pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
pip install transformers
pip install -U huggingface_hub
pip install accelerate -U
pip install tensorboard
```

### Train model
```sh
python trainModel.py --addNoise --augmentData --epochs=10 --nHiddenLayers=3 --image_resolution=224 --batch_size=64  --modelName=tfModelTones_v8

python fineTuneModel.py --addNoise --augmentData --unfreezeLastBaseLayer --epochs=2 --batch_size=64 --modelName=fineTunedModelTones_v1
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
aws s3 sync s3://chinesetonesdata/noise_data noise_data



