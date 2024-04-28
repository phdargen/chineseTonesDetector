import React, { useState, useEffect } from 'react';
import RecordRTC from 'recordrtc';
import AudioPlayer from 'react-audio-player';
import { Box, Button, Paper, Typography } from '@mui/material';

// API
// const api_url = 'http://localhost:5000/api/'
const api_url = 'https://8q3aqjs3v1.execute-api.us-east-2.amazonaws.com/prod/api/'

// Maximum recording time in seconds
const MAX_RECORDING_TIME = 1; 

const VoiceRecordingButton = () => {

  // Get example word to speak
  const [currentFile, setCurrentFile] = useState(null);
  const [soundInfo, setSoundInfo] = useState({ sound: '', tone: '', speaker: '' });
  const [audioPlayer, setAudioPlayer] = useState(new Audio());
  const [isPlaying, setIsPlaying] = useState(false);
  const [statistics, setStatistics] = useState({ correct: 0, total: 0 });
  const [soundBlob, setSoundBlob] = useState(null);

  useEffect(() => {
    return () => audioPlayer.pause();
  }, [audioPlayer]);

  const playRandomSound = () => {
    setIsPlaying(true);
    setSpectrumImage(null);
    setPrediction(null);
    setSoundBlob(null);
  
    fetch(api_url+'random-sound')
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then(data => {
        const { sound, tone, speaker, url } = data;
        setSoundInfo({ sound, tone, speaker });
        setCurrentFile(url);
        audioPlayer.src = url;
        audioPlayer.play().catch(e => console.error("Error playing audio:", e));
  
        // Fetch the audio blob from the URL provided by the API
        return fetch(url);  // This fetches the audio file as a response stream
      })
      .then(response => {
        if (!response.ok) {
          throw new Error('Failed to fetch audio blob for spectrum analysis');
        }
        return response.blob();  // Convert the response stream to a Blob
      })
      .then(blob => {
        setSoundBlob(blob);  // Save the blob if needed for other purposes
        //fetchSpectrum(blob);  // Now we call fetchSpectrum with the Blob
      })
      .catch(error => {
        console.error('Error fetching or playing sound:', error);
        setIsPlaying(false);
      });
  };  

  const replaySound = () => {

    if (audioPlayer.src) {
      audioPlayer.currentTime = 0; 
      if(audioPlayer.src !== currentFile ){
        audioPlayer.src = currentFile;
        audioPlayer.load();
      }
      audioPlayer.play().catch(e => console.error("Error replaying audio:", e));

      fetchSpectrum(soundBlob);
    }
  };

  // Record speaking
  const [recorder, setRecorder] = useState(null);
  const [audioBlob, setAudioBlob] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordedTime, setRecordedTime] = useState(0);
  const [spectrumImage, setSpectrumImage] = useState(null);
  const [prediction, setPrediction] = useState(null);

  useEffect(() => {
    let interval;
    if (isRecording) {
      interval = setInterval(() => {
        setRecordedTime((prevTime) => prevTime + 1);

        // Stop recording after MAX_RECORDING_TIME time
        if (recordedTime >= MAX_RECORDING_TIME) {
          stopRecording();
        }
      }, 1000);
    } else {
      clearInterval(interval);
    }

    return () => clearInterval(interval);
  }, [isRecording, recordedTime]);

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const options = { type: 'audio', mimeType: 'audio/webm' };
    const audioRecorder = new RecordRTC(stream, options);
    audioRecorder.startRecording();
    setRecorder(audioRecorder);
    setIsRecording(true);
    setRecordedTime(0);
  };

  const stopRecording = () => {
    if (recorder) {
      recorder.stopRecording(() => {
        const audioBlob = recorder.getBlob();
        setAudioBlob(audioBlob);
        setIsRecording(false);

        // Fetch spectrum image when recording stops
        fetchSpectrum(audioBlob);
      });
    }
  };

  const fetchSpectrum = async (audioData) => {
    try {
      const formData = new FormData();
      formData.append('audio', audioData); 
      console.log('audioData ', audioData)
      console.log('formData ', formData)

      const response = await fetch(api_url+"get_spectrum", {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setSpectrumImage(data.spectrum);
        setPrediction(data.prediction);
      } else {
        console.error('Failed to fetch spectrum data.');
      }
    } catch (error) {
      console.error('Error fetching spectrum data:', error);
    }
  };

  const replayAudio = () => {
    if (audioBlob) {
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      audio.play();
    }
  };

  const progressBarStyle = {
    width: `${(recordedTime / MAX_RECORDING_TIME) * 100}%`,
    background: 'linear-gradient(to right, #00aaff, #0077dd)',
    height: '20px', 
  };

  const getPredictionColor = () => {
    if( String(prediction) === soundInfo.tone ) return "green";
    return "red";
  };

return (
  <div>
    <Typography variant="h1">Speaking</Typography>

    <Button variant="contained" color="primary" onClick={playRandomSound} disabled={isPlaying && prediction === null}>
        Next Word
    </Button>
    <Button variant="contained" color="primary" onClick={replaySound} disabled={!currentFile}>
        Play
    </Button>
    
    <Box mb={2}>
          <Paper elevation={3} style={{ padding: '16px' }}>
            <Typography variant="h6"> Word: {soundInfo.sound} </Typography>
            <Typography variant="h6"> Tone: {soundInfo.tone} </Typography>
    </Paper>
    </Box>

    <Button variant="contained" color="primary" onClick={startRecording} disabled={isRecording}>
      Start Recording
    </Button>
    {/* <Button variant="contained" color="primary" onClick={stopRecording} disabled={!isRecording}>Stop Recording</Button> */}

    <Box mb={2}>
    <Paper elevation={3} style={{ padding: '16px' }}>
      {audioBlob && (
          <div>
            <AudioPlayer src={URL.createObjectURL(audioBlob)} controls />
          </div>
      )}

      { prediction && <Typography variant="h6" color={getPredictionColor()} > Prediction: Tone {prediction} </Typography> }

    </Paper>
    </Box>

    {spectrumImage && (
        <div>
          <img src={`data:image/png;base64,${spectrumImage}`} alt="Spectrum" />
        </div>
      )}

  </div>
);
          
};

export default VoiceRecordingButton;
