import React from 'react';
import { useTheme, useMediaQuery } from '@mui/material';
import { Grid, Box, Paper, Button, Typography } from '@mui/material';
import { Link } from 'react-router-dom';

import HearingIcon from '@mui/icons-material/Hearing';
import RecordVoiceOverIcon from '@mui/icons-material/RecordVoiceOver';
import SchoolIcon from '@mui/icons-material/School';
import LibraryBooksIcon from '@mui/icons-material/LibraryBooks';

const Home = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  
  return (

    <div className="home" style={{ padding: isMobile ? '10px' : '20px', margin: isMobile ? '10px' : '20px', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', textAlign: 'center', color: 'white' }}>

    <Typography variant={isMobile ? 'h4' : 'h2'} style={{ fontWeight: 'bold' }}> <br />  Pingulino <br /> <br /> </Typography>

    <Typography variant={isMobile ? 'h6' : 'h4'} style={{ fontWeight: 'bold' }}> Master Mandarin Chinese tone pronunciation with this free-to-use, open-source app powered by deep-learning algorithms. <br /> <br />  </Typography>

    <div style={{ display: 'flex', justifyContent: 'center', gap: '10px', flexDirection: isMobile ? 'column' : 'row' }}>
        <Button variant="contained" color="primary" startIcon={<SchoolIcon />} component={Link} to="/about">
          Learn more
        </Button>
        <Button variant="contained" color="primary" startIcon={<HearingIcon />} component={Link} to="/listening">
          Practice listening
        </Button>
        <Button variant="contained" color="primary" startIcon={<RecordVoiceOverIcon />} component={Link} to="/speaking">
          Practice speaking
        </Button>
      </div>

    </div>
  );
};

export default Home;