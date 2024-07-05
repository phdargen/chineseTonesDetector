# ChineseTonesDetector

Machine learning project to help practice correct Mandarin Chinese tone pronunciation.

The [Tone Perfect](https://tone.lib.msu.edu/)
 dataset from Michigan State University is used for the training.
It includes the full catalog of monosyllabic sounds in Mandarin Chinese in all four tones, spoken by six native Mandarin speakers.
The collection is comprised of about 10k samples with a total duration of approximately 2 hours.
To create a more diverse and realistic training sample, the original dataset is augmented.
The audio samples are then converted to Mel spectrograms as input for image classification algorithms. 

A CNN and a fine-tuned Vision Transformer (based on Google's [vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)) model are trained.
Both achieve an accuracy of >99.9% on a statistically independent test dataset.
More details and deployed models for inference can be found here: https://pingulino.vercel.app/

## Example spectrograms

![ma1_FV1](https://github.com/phdargen/chineseTonesDetector/assets/29732335/e3906d99-6bf0-4e13-9c80-70fadc1d3238)
![ma2_FV1](https://github.com/phdargen/chineseTonesDetector/assets/29732335/9cd972f7-3ca1-4ecb-9830-cd00264e1940)

![ma3_FV1](https://github.com/phdargen/chineseTonesDetector/assets/29732335/64657bdf-9dbc-424b-925b-c85bca40dca1)
![ma4_FV1](https://github.com/phdargen/chineseTonesDetector/assets/29732335/5c210d00-b805-4a6d-a767-519ad09d93d8)

## Train ML

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
pip install peft
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
python fineTuneModel.py --addNoise --augmentData --epochs=1 --batch_size=64 --modelName=fineTunedModelTonesLora_v1 --applyLora
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
### get local copy of base ViT model
```sh
sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/google/vit-base-patch16-224
```

### run
```sh
gunicorn --workers 3 --bind 0.0.0.0:5000 spectrum:app
```

## Generate vocabulary list for anki
```sh
pip install pypinyin genanki 
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



