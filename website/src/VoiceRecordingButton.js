import React, { useState, useEffect } from 'react';
import RecordRTC from 'recordrtc';
import AudioPlayer from 'react-audio-player';

// Maximum recording time in seconds
const MAX_RECORDING_TIME = 2; 

const VoiceRecordingButton = () => {
  const [recorder, setRecorder] = useState(null);
  const [audioBlob, setAudioBlob] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordedTime, setRecordedTime] = useState(0);
  const [spectrumImage, setSpectrumImage] = useState(null);

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
  
      const response = await fetch('http://localhost:5000/get_spectrum', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setSpectrumImage(data.spectrum);
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

  return (
    <div>
      <button onClick={startRecording} disabled={isRecording}>
        Start Recording
      </button>
      {isRecording ? (
        <button onClick={stopRecording}>Stop Recording</button>
      ) : (
        audioBlob && (
          <div>
            <p>Recorded Audio:</p>
            <AudioPlayer src={URL.createObjectURL(audioBlob)} controls />
            <button onClick={replayAudio}>Replay Audio</button>
          </div>
        )
      )}
      {isRecording && (
        <div>
          <p>Recording Time: {recordedTime} seconds</p>
        </div>
      )}
      {spectrumImage && (
        <div>
          <img src={`data:image/png;base64,${spectrumImage}`} alt="Spectrum" />
        </div>
      )}
    </div>
  );
};

export default VoiceRecordingButton;
