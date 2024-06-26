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

pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
pip install transformers
pip install -U huggingface_hub
pip install accelerate -U
pip install tensorboard
```

### Download data samples
```sh
cd prepareData
python downloadTonesData.py
```

### Notebook to explore data samples
```sh
cd prepareData
analyzeTones.ipynb
```

### Train CNN model
```sh
cd trainML
python trainModel.py --addNoise --augmentData --epochs=10 --nHiddenLayers=3 --image_resolution=128 --batch_size=64 --modelName=tfModelTones_v8
```

### Train fine-tuned ViT model
```sh
cd trainML
python fineTuneModel.py --addNoise --augmentData --unfreezeLastBaseLayer --epochs=10 --batch_size=64 --modelName=fineTunedModelTones_v1
```

## ML API (local)

### setup environment
```sh
pip install librosa Flask flask-cors pydub
(brew install ffmpeg)
pip install torch torchvision torchaudio
pip install transformers
```

### run
```sh
cd flaskAPI
python spectrum.py
```

## ML API (AWS EC2 Ubuntu)

### setup environment
```sh
sudo apt-get update
sudo apt-get install ffmpeg libavcodec-extra
sudo apt  install emacs
sudo apt  install tmux
pip install requests numpy matplotlib pandas seaborn boto3
pip install  librosa
TMPDIR=~/tmp/ pip install tensorflow
pip install Flask flask-cors pydub
pip install gunicorn
```
### run
```sh
gunicorn --workers 3 --bind 0.0.0.0:5000 spectrum:app
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



