import React from 'react';
import { Typography, useTheme, useMediaQuery } from '@mui/material';

function Header() {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  const headerStyle = {
    display: 'flex',
    alignItems: 'center',
    padding: '10px',
    backgroundColor: '#3f51b5',
    marginRight: '0px',
    backgroundImage: 'linear-gradient(to right, #3f51b5, #5a55ae)', // Gradient from dark blue to lighter blue
  };

  const logoStyle = {
    width: isMobile ? '100px' : '150px', 
    height: 'auto',
    marginRight: '20px',
  };

  const typographyStyle = {
    fontWeight: 'bold',
    color: 'white',
    // fontSize: isMobile ? '1.5rem' : '2.5rem', 
  };

  return (
    <div style={headerStyle}>
      {/* <img src="pingulino2.png" alt="Logo" style={logoStyle} /> */}
      <Typography variant={isMobile ? "h3" : "h1"} style={typographyStyle}>
        Pingulino
      </Typography>
    </div>
  );
}

export default Header;
