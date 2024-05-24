import React, { useState, useEffect } from "react";
import { Box, List, Grid, ListItemIcon, ListItemButton, ListItemText, Typography, Link } from '@mui/material';
import { useTheme, useMediaQuery } from '@mui/material';
import { useLocation } from "react-router-dom";

import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faTwitterSquare, faGithubSquare, faInstagramSquare } from '@fortawesome/free-brands-svg-icons'
const gitLink = "https://github.com/phdargen/chineseTonesDetector"

function Sidebar() {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  const [selectedIndex, setSelectedIndex] = useState(0);
  const handleListItemClick = (
    index
  ) => {
    setSelectedIndex(index);
  };

  const location = useLocation();
  useEffect(() => {
    if (location.pathname === "/") setSelectedIndex(0);
    if (location.pathname === "/home") setSelectedIndex(0);
    if (location.pathname === "/listening") setSelectedIndex(1);
    if (location.pathname === "/speaking") setSelectedIndex(2);
    if (location.pathname === "/about") setSelectedIndex(3);
  }, [location]);

  return (
    <div className="sidebar" style={{ display: 'flex', flexDirection: 'column', height: '100vh', width: isMobile ? '30%' : '20%' }}>
      {/* <Typography variant="h2" align='center' style={{ fontWeight: 'bold', color: 'white' }} > Pingulino</Typography> */}
      
      <Box sx={{ padding: '5px' }}>
        <img 
          src="pingulino2.png" 
          style={{ width: '100%' }}
        />
      </Box>

      <List style={{ flexDirection: 'column', height: '100%' }}>
        <ListItemButton component={Link} to="/home" onClick={() => handleListItemClick(0)} sx={selectedIndex === 0? {border: 1 } : {}} >
          <ListItemText primary={<Typography variant={isMobile ? "h6" : "h5"} style={{ fontWeight: 'bold', color: 'white' }}> Home</Typography>}/>    
        </ListItemButton>
        <ListItemButton component={Link} to="/listening" onClick={() => handleListItemClick(1)} sx={selectedIndex === 1? {border: 1 } : {}} >
          <ListItemText primary={<Typography variant={isMobile ? "h6" : "h5"} style={{ fontWeight: 'bold', color: 'white' }}> Listening</Typography>}/>    
        </ListItemButton>
        <ListItemButton component={Link} to="/speaking" onClick={() => handleListItemClick(2)} sx={selectedIndex === 2? {border: 1 } : {}}>
         <ListItemText primary={<Typography variant={isMobile ? "h6" : "h5"} style={{ fontWeight: 'bold', color: 'white' }}> Speaking</Typography>}/>    
        </ListItemButton>
        <ListItemButton component={Link} to="/about" onClick={() => handleListItemClick(3)} sx={selectedIndex === 3? {border: 1 } : {}}>
          <ListItemText primary={<Typography variant={isMobile ? "h6" : "h5"} style={{ fontWeight: 'bold', color: 'white' }}> About</Typography>}/>    
        </ListItemButton>
      </List>

      <Box textAlign="center" pt={{ xs: 5, sm: 10 }} pb={{ xs: 5, sm: 0 }} sx={{ padding: '5px' }}>
        <Link href={gitLink} target="_blank" rel="noopener noreferrer">
          <FontAwesomeIcon icon={faGithubSquare} size="3x" color='white'  />
        </Link>
        <Typography variant="body2" color="white" pt={2}>
          {'© '} Pingulino {' '} {new Date().getFullYear()}
        </Typography>
      </Box>


    </div>
  );
}

export default Sidebar;
