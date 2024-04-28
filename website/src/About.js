import React, { useState, useEffect } from 'react';
import { useTheme, useMediaQuery } from '@mui/material';

const About = () => {

  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  const [aboutText, setAboutText] = useState('');

  useEffect(() => {
    fetch('https://chinese-tones-detector-ml-api.vercel.app/about')  
      .then(response => response.text())
      .then(text => setAboutText(text))
      .catch(err => console.error('Failed to fetch About data:', err));
  }, []);

  return <h1>{aboutText || 'About'}</h1>;
};

export default About;
