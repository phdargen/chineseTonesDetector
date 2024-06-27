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
          <Typography variant={isMobile ? 'h6' : 'h4'} style={{ fontWeight: 'bold' }}> The Importance of Tones in Mandarin Chinese</Typography>

          <Typography variant={isMobile ? 'body2' : 'body1'} style={{ marginTop: '16px' }}>
            Mandarin Chinese is a tonal language, where the pitch of a word changes its meaning completely. <br />
            For example, "ma" can mean "mother," "hemp," "horse," or "to scold," depending on the tone:<br />
            1. <strong>First Tone</strong>: High and level (mā - 妈, "mother")<br />
            2. <strong>Second Tone</strong>: Rising (má - 麻, "hemp")<br />
            3. <strong>Third Tone</strong>: Falling and rising (mǎ - 马, "horse")<br />
            4. <strong>Fourth Tone</strong>: Sharp and falling (mà - 骂, "to scold")<br />
            This is different from English, where tone mainly conveys emotion or emphasis. <br />
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
                  <Typography variant={isMobile ? 'h6' : 'h4'} style={{ fontWeight: 'bold' }}> 妈 (mā)</Typography>
                  <Typography variant={isMobile ? 'body2' : 'body1'} >mother</Typography>
                  <AudioPlayer src='ma1_FV1_MP3.mp3' controls className="audioPlayer" />
                </Box>   
                <img src='ma1_FV1.png' style={{ width: '100%', marginBottom: '16px' }} />
                <Typography variant={isMobile ? 'body2' : 'body1'} >1st tone: High and level</Typography>
                <Typography variant={isMobile ? 'body2' : 'body1'} > </Typography>

              </Paper>
            </Box>
          </Grid>

          <Grid item xs={12} sm={6} >
            <Box mb={2} width={isMobile ? '100%' : '100%'}>
              <Paper elevation={3} style={{ padding: '16px' }}>
                <Box display="flex" flexDirection="column" alignItems="center" mt={2}>
                  <Typography variant={isMobile ? 'h6' : 'h4'} style={{ fontWeight: 'bold' }}>麻 (má)</Typography>
                  <Typography variant={isMobile ? 'body2' : 'body1'} >hemp</Typography>
                  <AudioPlayer src='ma2_FV1_MP3.mp3' controls className="audioPlayer" />
                </Box>   
                <img src='ma2_FV1.png' style={{ width: '100%', marginBottom: '16px' }} />
                <Typography variant={isMobile ? 'body2' : 'body1'} >2nd tone: Rising</Typography>
              </Paper>
            </Box>
          </Grid>

          <Grid item xs={12} sm={6} >
            <Box mb={2} width={isMobile ? '100%' : '100%'}>
              <Paper elevation={3} style={{ padding: '16px' }}>
                <Box display="flex" flexDirection="column" alignItems="center" mt={2}>
                  <Typography variant={isMobile ? 'h6' : 'h4'} style={{ fontWeight: 'bold' }}>马 (mǎ)</Typography>
                  <Typography variant={isMobile ? 'body2' : 'body1'} >horse</Typography>
                  <AudioPlayer src='ma3_FV1_MP3.mp3' controls className="audioPlayer" />
                </Box>   
                <img src='ma3_FV1.png' style={{ width: '100%', marginBottom: '16px' }} />
                <Typography variant={isMobile ? 'body2' : 'body1'} >3rd tone:  Falling and rising</Typography>
              </Paper>
            </Box>
          </Grid>

          <Grid item xs={12} sm={6} >
            <Box mb={2} width={isMobile ? '100%' : '100%'}>
              <Paper elevation={3} style={{ padding: '16px' }}>
                <Box display="flex" flexDirection="column" alignItems="center" mt={2}>
                  <Typography variant={isMobile ? 'h6' : 'h4'} style={{ fontWeight: 'bold' }}>骂 (mà)</Typography>
                  <Typography variant={isMobile ? 'body2' : 'body1'} >to scold or to curse</Typography>
                  <AudioPlayer src='ma4_FV1_MP3.mp3' controls className="audioPlayer" />
                </Box>   
                <img src='ma4_FV1.png' style={{ width: '100%', marginBottom: '16px' }} />
                <Typography variant={isMobile ? 'body2' : 'body1'} >4th tone: Sharp and falling</Typography>
              </Paper>
            </Box>
          </Grid>

      </Grid>

      <Grid item xs={12} sm={6} >
          <Box mb={2} width={isMobile ? '100%' : '100%'}>
          <Paper elevation={3} style={{ padding: '16px' }}>
          <Typography variant={isMobile ? 'h6' : 'h4'} style={{ fontWeight: 'bold' }}> Mel Spectrograms</Typography>

          <Typography variant={isMobile ? 'body2' : 'body1'} style={{ marginTop: '16px' }}>
          Spectrograms are generated from sound signals using fourier transforms, which decompose the signal into its constituent frequencies and display the amplitude of each frequency present. 
          A mel spectrogram visually represents sound, showing how it changes over time. 
          The x-axis represents time, displaying the progression of the sound from start to finish. 
          The y-axis represents frequency (or pitch) on the mel scale, which adjusts frequencies to match how humans perceive pitch. 
          This adjustment makes it easier to see the distinct pitch patterns of Mandarin Chinese tones. 
          The color bar shows the sound's loudness (amplitude) using the decibel (dB) scale, where louder sounds are brighter.
          The decibel (dB) scale is logarithmic, meaning each 10 dB increase represents a tenfold increase in sound intensity and is perceived by humans as roughly twice as loud.
          Pingulino aids learners in mastering Mandarin tones by providing a clear visual guide to pronunciation.

        </Typography>

          </Paper>
          </Box>
      </Grid>

      <Grid item xs={12} sm={6} >
          <Box mb={2} width={isMobile ? '100%' : '100%'}>
          <Paper elevation={3} style={{ padding: '16px' }}>
          <Typography variant={isMobile ? 'h6' : 'h4'} style={{ fontWeight: 'bold' }}> Machine Learning </Typography>

          <Typography variant={isMobile ? 'body2' : 'body1'} style={{ marginTop: '16px' }}>
          Pingulino provides instant visual feedback on your pronunciation by displaying the mel spectrogram of your voice recording. 
          Additionally, machine learning algorithms predict the tone of your recording. Instead of using audio samples directly, mel spectrograms are employed to train image classification algorithms. 
          By analyzing the visual patterns in mel spectrograms, these machine learning models learn to recognize the correct pitch patterns for each tone. When you speak into the system, it compares your pronunciation to the ideal patterns using these trained models.        
          <br />  <br />
          Standard speech recognition models typically focus on transcribing speech to produce valid language from audio input. 
          For instance, take the phrase "nǐ hǎo," which means "hello" in Mandarin.
          If you pronounce it as "Nì hào," a conventional Mandarin speech recognition model would likely correct it to "nǐ hǎo," aiming to understand and produce valid Mandarin, rather than providing feedback on pronunciation accuracy.
          In contrast, Pingulino's machine learning models are specifically designed to distinguish between different tones, ensuring that users receive precise feedback on their pronunciation.
          </Typography>

          </Paper>
          </Box>
      </Grid>

      <Grid item xs={12} sm={6} >
          <Box mb={2} width={isMobile ? '100%' : '100%'}>
          <Paper elevation={3} style={{ padding: '16px' }}>
          <Typography variant={isMobile ? 'h6' : 'h4'} style={{ fontWeight: 'bold' }}> Dataset </Typography>

          <Typography variant={isMobile ? 'body2' : 'body1'} style={{ marginTop: '16px' }}>
          Pingulino's machine learning algorithms are trained using the 'Tone Perfect' dataset from Michigan State University. 
          This dataset includes the full catalog of monosyllabic sounds in Mandarin Chinese in all four tones. 
          Spoken by six native Mandarin speakers (three female and three male), the collection is comprised of about 10k audio files with a total duration of approximately 2 hours.
          <br /> <br />
          [Catherine Ryu, Mandarin Tone Perception & Production Team, and Michigan State University Libraries. <a href="https://tone.lib.msu.edu/" target="_blank" rel="noopener noreferrer">Tone Perfect: Multimodal Database for Mandarin Chinese.</a> ]
          </Typography>

          <Typography variant={isMobile ? 'body2' : 'body1'} style={{ marginTop: '16px' }}>
          Audio samples from the Tone Perfect dataset are standardized, high-quality, and from a limited set of speakers. 
          To create a more diverse and realistic training sample, 
          the original dataset is augmented by altering the audio speed without changing the pitch, 
          shifting the audio forward or backward in time, changing the pitch without affecting speed, 
          introducing random noise to simulate real-world conditions, adjusting the volume, 
          and applying frequency and time masking to the audio's spectrogram. 
          <br /> 
          </Typography>

          </Paper>
          </Box>
      </Grid>

      <Grid item xs={12} sm={6} >
          <Box mb={2} width={isMobile ? '100%' : '100%'}>
          <Paper elevation={3} style={{ padding: '16px' }}>
          <Typography variant={isMobile ? 'h6' : 'h4'} style={{ fontWeight: 'bold' }}> Classification Models </Typography>

          <Typography variant={isMobile ? 'body2' : 'body1'} style={{ marginTop: '16px' }}>

          Pingulino employs two machine learning approaches for analyzing mel spectrograms: 
          a Convolutional Neural Network (CNN) and a fine-tuned version of Google's Vision Transformer (ViT), 
          specifically the <a href="https://huggingface.co/google/vit-base-patch16-224" target="_blank" rel="noopener noreferrer">'vit-base-patch16-224'</a> model.
          The CNN, inspired by the human visual cortex, excels at identifying local patterns in images. 
          It uses sliding filters to detect features, making it efficient for straightforward tone classification.
          In contrast, the ViT model treats 
          images as sequences of patches, processing them similarly to text. 
          It leverages transfer learning to capture complex, global relationships in spectrograms. 
          Both models achieve an accuracy of over 99.9% on a statistically independent test dataset.
          </Typography>

          </Paper>
          </Box>
      </Grid>

      <Grid item xs={12} sm={6} >
          <Box mb={2} width={isMobile ? '100%' : '100%'}>
          <Paper elevation={3} style={{ padding: '16px' }}>
          <Typography variant={isMobile ? 'h6' : 'h4'} style={{ fontWeight: 'bold' }}> Privacy Disclaimer </Typography>

          <Typography variant={isMobile ? 'body2' : 'body1'} style={{ marginTop: '16px' }}>
          We use Vercel Web Analytics to collect anonymized data about your visit to our site, including pages visited, operating system, and browser. 
          By using our site, you consent to this data collection.
          No personal information or identifiers that track users across different sites are collected or stored. 
          Audio recordings are not stored and are only used for interference in our machine learning models.
          </Typography>

          </Paper>
          </Box>
      </Grid>

    </div>
  );
};

export default About;
