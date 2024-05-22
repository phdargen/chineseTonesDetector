import React, { useState, useEffect, useRef } from 'react';
import { useTheme, useMediaQuery } from '@mui/material';
import { Grid, Box, Paper, Button, Typography } from '@mui/material';
import PlayCircleFilledRoundedIcon from '@mui/icons-material/PlayCircleFilledRounded';
import AudioPlayer from 'react-audio-player';

const About = () => {

  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  
  return (

    <div className="about" style={{ padding: isMobile ? '10px' : '20px', margin: isMobile ? '10px' : '20px', justifyContent: 'center', alignItems: 'center'  }}>

    <Grid item xs={12} sm={6} >
          <Box mb={2} width={isMobile ? '100%' : '100%'}>
          <Paper elevation={3} style={{ padding: '16px' }}>
          <Typography variant="h4" style={{ fontWeight: 'bold' }}> The Importance of Tones in Mandarin Chinese</Typography>

          <Typography variant="body1" style={{ marginTop: '16px' }}>
        Mandarin Chinese is a tonal language, where the pitch of a word can change its meaning completely. 
        Correctly using tones in Mandarin is crucial for clear communication. 
        Mandarin has four main tones:<br />
        1. <strong>First Tone</strong>: High and level (mā - 妈, "mother")<br />
        2. <strong>Second Tone</strong>: Rising (má - 麻, "hemp")<br />
        3. <strong>Third Tone</strong>: Falling and rising (mǎ - 马, "horse")<br />
        4. <strong>Fourth Tone</strong>: Sharp and falling (mà - 骂, "to scold")<br />
        For example, "ma" can mean "mother," "hemp," "horse," or "to scold," depending on the tone.<br />

        This is different from English, where tone mainly conveys emotion or emphasis.
        For instance, "Really?" (rising tone) indicates a question or surprise, while "I really like it" (emphasis on "really") shows strong affirmation.
        </Typography>

          </Paper>
          </Box>
    </Grid>

    <Grid container spacing={2}>

          <Grid item xs={12} sm={6} >
            <Box mb={2} width={isMobile ? '100%' : '100%'}>
              <Paper elevation={3} style={{ padding: '16px' }}>
                <Box display="flex" flexDirection="column" alignItems="center" mt={2}>
                  <Typography variant="h4" style={{ fontWeight: 'bold' }}> 妈 (mā)</Typography>
                  <Typography variant="h6" >mother</Typography>
                  <AudioPlayer src='ma1_FV1_MP3.mp3' controls className="audioPlayer" />
                </Box>   
                <img src='ma1_FV1.png' style={{ width: '100%', marginBottom: '16px' }} />
                <Typography variant="h6" >1st tone: High and level</Typography>
                <Typography variant="h6" > </Typography>

              </Paper>
            </Box>
          </Grid>

          <Grid item xs={12} sm={6} >
            <Box mb={2} width={isMobile ? '100%' : '100%'}>
              <Paper elevation={3} style={{ padding: '16px' }}>
                <Box display="flex" flexDirection="column" alignItems="center" mt={2}>
                  <Typography variant="h4" style={{ fontWeight: 'bold' }}>麻 (má)</Typography>
                  <Typography variant="h6" >hemp</Typography>
                  <AudioPlayer src='ma2_FV1_MP3.mp3' controls className="audioPlayer" />
                </Box>   
                <img src='ma2_FV1.png' style={{ width: '100%', marginBottom: '16px' }} />
                <Typography variant="h6" >2nd tone: Rising</Typography>
              </Paper>
            </Box>
          </Grid>

          <Grid item xs={12} sm={6} >
            <Box mb={2} width={isMobile ? '100%' : '100%'}>
              <Paper elevation={3} style={{ padding: '16px' }}>
                <Box display="flex" flexDirection="column" alignItems="center" mt={2}>
                  <Typography variant="h4" style={{ fontWeight: 'bold' }}>马 (mǎ)</Typography>
                  <Typography variant="h6" >horse</Typography>
                  <AudioPlayer src='ma3_FV1_MP3.mp3' controls className="audioPlayer" />
                </Box>   
                <img src='ma3_FV1.png' style={{ width: '100%', marginBottom: '16px' }} />
                <Typography variant="h6" >3rd tone:  Falling and rising</Typography>
              </Paper>
            </Box>
          </Grid>

          <Grid item xs={12} sm={6} >
            <Box mb={2} width={isMobile ? '100%' : '100%'}>
              <Paper elevation={3} style={{ padding: '16px' }}>
                <Box display="flex" flexDirection="column" alignItems="center" mt={2}>
                  <Typography variant="h4" style={{ fontWeight: 'bold' }}>骂 (mà)</Typography>
                  <Typography variant="h6" >to scold or to curse</Typography>
                  <AudioPlayer src='ma4_FV1_MP3.mp3' controls className="audioPlayer" />
                </Box>   
                <img src='ma4_FV1.png' style={{ width: '100%', marginBottom: '16px' }} />
                <Typography variant="h6" >4th tone: Sharp and falling</Typography>
              </Paper>
            </Box>
          </Grid>

      </Grid>

    </div>
  );
};

export default About;
