import React, { useState, useEffect } from 'react';
import { Box, Button, Paper, Typography } from '@mui/material';

const speakers = ["MV1", "MV2", "MV3", "FV1", "FV2"];
function getNewSpeaker(speakers, currentSpeaker) {
  // Filter out the current speaker from the speakers array
  const filteredSpeakers = speakers.filter(speaker => speaker !== currentSpeaker);
  const randomIndex = Math.floor(Math.random() * filteredSpeakers.length);
  return filteredSpeakers[randomIndex];
}

// API
// const api_url = 'http://localhost:5000/api/'
const api_url = 'https://8q3aqjs3v1.execute-api.us-east-2.amazonaws.com/prod/api/'

async function fetchUrl(newFileUrl) {
  try {
      // Fetch URL 
      const apiUrl = api_url+`get_file_access?filename=${encodeURIComponent(newFileUrl)}`;
      const response = await fetch(apiUrl);
      const data = await response.json();

      if (response.ok) {
          console.log('New URL: ', data.url);
          return data.url;
      } else {
          console.error('Failed to fetch presigned URL:', data.error);
          return null;
      }
  } catch (error) {
      console.error('Error fetching presigned URL:', error);
      return null;
  }
}

function Listening() {
  const [currentFile, setCurrentFile] = useState(null);
  const [soundInfo, setSoundInfo] = useState({ sound: '', tone: '', speaker: '' });
  const [audioPlayer, setAudioPlayer] = useState(new Audio());
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedTone, setSelectedTone] = useState(null);
  const [statistics, setStatistics] = useState({ correct: 0, total: 0 });

  useEffect(() => {
    return () => audioPlayer.pause();
  }, [audioPlayer]);

  const playRandomSound = () => {
    setIsPlaying(true);
    setSelectedTone(null);
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
        audioPlayer.src = url;
        audioPlayer.play().catch(e => console.error("Error playing audio:", e));
        setCurrentFile(url);
      })
      .catch(error => {
        console.error('Error playing sound:', error);
        setIsPlaying(false);
      });
  };

  const handlePlayTone = async (newTone) => {
    if (!currentFile) {
      console.error("No audio file loaded");
      return;
    }

    if (audioPlayer.src) {
      audioPlayer.pause();
      audioPlayer.currentTime = 0;  
    }

    // Extract parts of the filename
    const filePattern = /(\w+)(\d+)_(\w+)_MP3\.mp3/;
    const match = filePattern.exec(currentFile);
  
    if (match) {
      const [fullMatch, sound, tone, speaker] = match;

      // Construct URL with new tone
      const newFileName = `${sound}${newTone}_${speaker}_MP3.mp3`; 
      console.log('newFileName ', newFileName);
      const newFileUrl = await fetchUrl(newFileName);
      if (newFileUrl) {
        console.log('newFileUrl ', newFileUrl);
        audioPlayer.src = newFileUrl;
        audioPlayer.load();  
        audioPlayer.play().catch(e => console.error("Error playing tone:", e));
      } else {
        console.log('Failed to load URL');
      }
    } else {
      console.error("Failed to parse current file URL:", currentFile);
    }
  };

  const handleDifferentSpeaker = async () => {
    if (!currentFile) {
      console.error("No audio file loaded");
      return;
    }

    if (audioPlayer.src) {
      audioPlayer.pause();
      audioPlayer.currentTime = 0;  
    }

    // Extract parts of the filename
    const filePattern = /(\w+)(\d+)_(\w+)_MP3\.mp3/;
    const match = filePattern.exec(currentFile);
  
    if (match) {
      const [fullMatch, sound, tone, speaker] = match;
      const newSpeaker = getNewSpeaker(speakers, speaker);

      // Construct URL with new speaker
      const newFileName = `${sound}${tone}_${newSpeaker}_MP3.mp3`; 
      console.log('newFileName ', newFileName);
      const newFileUrl = await fetchUrl(newFileName);
      if (newFileUrl) {
        console.log('newFileUrl ', newFileUrl);
        audioPlayer.src = newFileUrl;
        audioPlayer.load();  
      } else {
        console.log('Failed to load URL');
      }

      var success = true;
      audioPlayer.play().catch(e => { 
        console.error("Error changing speaker:", e);
        audioPlayer.src = currentFile;
        audioPlayer.load();
        audioPlayer.play().catch(e => console.error("Error playing audio:", e));
        success = false;
      });
      
      if(success){
        setSoundInfo({ sound, tone, speaker: newSpeaker });
        setCurrentFile(newFileUrl);
      }
    } else {
      console.error("Failed to parse current file URL:", currentFile);
    }

  };

  const replaySound = () => {

    if (audioPlayer.src) {
      audioPlayer.currentTime = 0; 
      if(audioPlayer.src !== currentFile ){
        audioPlayer.src = currentFile;
        audioPlayer.load();
      }
      audioPlayer.play().catch(e => console.error("Error replaying audio:", e));
    }
  };

  const handleToneSelection = (tone) => {
    setSelectedTone(tone);
    setStatistics(prev => ({
      correct: prev.correct + (tone === soundInfo.tone ? 1 : 0),
      total: prev.total + 1
    }));
    setIsPlaying(false);
  };

  const getButtonColor = (tone) => {
    if(tone === soundInfo.tone ) return "green";
    if(tone === selectedTone && tone !== soundInfo.tone ) return "red";
    return "primary";
  };

  return (
    <div className="listening">
      <Typography variant="h1">Listening</Typography>
      <Button variant="contained" color="primary" onClick={playRandomSound} disabled={isPlaying}>
        Next Word
      </Button>
      <Button variant="contained" color="primary" onClick={replaySound} disabled={!currentFile}>
        Replay
      </Button>
      <Button variant="contained" color="primary" onClick={handleDifferentSpeaker} disabled={!currentFile}>
        Change Speaker
      </Button>
      <div>

        <Box mb={2}>
          <Paper elevation={3} style={{ padding: '16px' }}>
            <Typography variant="h6"> {soundInfo.sound}</Typography>
            {/* <Typography variant="h6">Tone: {soundInfo.tone}</Typography> */}
            {/* <Typography variant="h6">Speaker: {soundInfo.speaker}</Typography> */}

            <Typography variant="h6">
              Select Tone: {' '}
              {['1', '2', '3', '4'].map(tone => (
                <Button
                    key={tone}
                    onClick={() => handleToneSelection(tone)}
                    disabled={selectedTone !== null || currentFile === null}
                    sx={{
                      bgcolor: 'primary',  
                      color: 'primary',  
                      ':hover': {
                        bgcolor: 'primary',  
                      },
                      '&.Mui-disabled': {
                        bgcolor: 'primary',  
                        color: getButtonColor(tone),  
                      }
                    }}
                  >
                    Tone {tone}
                  </Button>
              ))}
            </Typography>

            <Typography variant="h6">
              Play Tone: {' '}
              {['1', '2', '3', '4'].map(tone => (
                <Button
                    key={tone}
                    onClick={() => handlePlayTone(tone)}
                    disabled={selectedTone === null || currentFile === null}
                  >
                    Tone {tone}
                  </Button>
              ))}
            </Typography>

          </Paper>
        </Box>

        <Box mb={2}>
          <Paper elevation={3} style={{ padding: '16px' }}>
          <Typography variant="h6">
              Statistics: {statistics.correct} / {statistics.total} (
             {statistics.total > 0 ? (statistics.correct / statistics.total * 100).toFixed(1) : 0} %)
          </Typography>
          </Paper>
        </Box>
      </div>
    </div>
  );
}

export default Listening;
