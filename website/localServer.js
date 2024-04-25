const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
const soundsDir = path.join(__dirname, '../raw_data');
const audioFilesDirectory = path.join(__dirname, '../raw_data');

const cors = require('cors');
const corsOptions = {
    origin: 'http://localhost:3000', 
    optionsSuccessStatus: 200
};
app.use(cors(corsOptions));
app.use('/sounds', cors(corsOptions), express.static(audioFilesDirectory));

app.get('/api/random-sound', (req, res) => {
  fs.readdir(soundsDir, (err, files) => {
    if (err) {
      return res.status(500).send('Failed to read sounds directory');
    }

    const mp3Files = files.filter(file => file.endsWith('.mp3'));
    const randomFile = mp3Files[Math.floor(Math.random() * mp3Files.length)];
    const match = /(\w+)(\d+)_(\w+)_MP3\.mp3/.exec(randomFile);

    if (match) {
      const [_, sound, tone, speaker] = match;

      res.json({
        sound,
        tone,
        speaker,
        url: `http://localhost:4000/sounds/${randomFile}` 
        //url: `${soundsDir}/${randomFile}` 
      });
    } else {
      res.status(404).send('No valid sound files found');
    }
  });
});

app.get('/', (req, res) => {
    res.send('Server is running and serving files');
  });

app.listen(4000, () => console.log('Server running on port 4000'));
