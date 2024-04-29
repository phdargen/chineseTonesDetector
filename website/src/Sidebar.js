import React, { useState, useEffect } from "react";
import { Box, List, ListItemIcon, ListItemButton, ListItemText, Typography, Link } from '@mui/material';
import { useTheme, useMediaQuery } from '@mui/material';
import { useLocation } from "react-router-dom";

import HeadsetMicRoundedIcon from '@mui/icons-material/HeadsetMicRounded';

function Sidebar() {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  const [selectedIndex, setSelectedIndex] = useState(1);
  const handleListItemClick = (
    index
  ) => {
    setSelectedIndex(index);
  };

  const location = useLocation();
  useEffect(() => {
    if (location.pathname === "/") setSelectedIndex(1);
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
        <ListItemButton component={Link} to="/listening" onClick={() => handleListItemClick(1)} sx={selectedIndex === 1? {border: 1 } : {}} >
          {/* <ListItemIcon>
            <HeadsetMicRoundedIcon style={{ color: 'white', padding: '0px', margin: '0px' }} /> 
          </ListItemIcon> */}
          <ListItemText primary={<Typography variant={isMobile ? "h6" : "h5"} style={{ fontWeight: 'bold', color: 'white' }}> Listening</Typography>}/>    
        </ListItemButton>
        <ListItemButton component={Link} to="/speaking" onClick={() => handleListItemClick(2)} sx={selectedIndex === 2? {border: 1 } : {}}>
         <ListItemText primary={<Typography variant={isMobile ? "h6" : "h5"} style={{ fontWeight: 'bold', color: 'white' }}> Speaking</Typography>}/>    
        </ListItemButton>
        <ListItemButton component={Link} to="/about" onClick={() => handleListItemClick(3)} sx={selectedIndex === 3? {border: 1 } : {}}>
          <ListItemText primary={<Typography variant={isMobile ? "h6" : "h5"} style={{ fontWeight: 'bold', color: 'white' }}> About</Typography>}/>    
        </ListItemButton>
      </List>
    </div>
  );
}

export default Sidebar;
