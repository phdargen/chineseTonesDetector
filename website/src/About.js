import React, { useState, useEffect } from 'react';
import { useTheme, useMediaQuery } from '@mui/material';

const About = () => {

  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  return <h1>{'About'}</h1>;
};

export default About;
