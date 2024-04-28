import React from 'react';
import { Box, List, ListItemIcon, ListItemButton, ListItemText, Typography, Link } from '@mui/material';
import { useTheme, useMediaQuery } from '@mui/material';

import HeadsetMicRoundedIcon from '@mui/icons-material/HeadsetMicRounded';

function Sidebar() {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

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
        <ListItemButton component={Link} to="/listening" >
          {/* <ListItemIcon>
            <HeadsetMicRoundedIcon style={{ color: 'white', padding: '0px', margin: '0px' }} /> 
          </ListItemIcon> */}
          <ListItemText primary={<Typography variant={isMobile ? "h6" : "h5"} style={{ fontWeight: 'bold', color: 'white' }}> Listening</Typography>}/>    
        </ListItemButton>
        <ListItemButton component={Link} to="/speaking">
         <ListItemText primary={<Typography variant={isMobile ? "h6" : "h5"} style={{ fontWeight: 'bold', color: 'white' }}> Speaking</Typography>}/>    
        </ListItemButton>
        <ListItemButton component={Link} to="/about">
          <ListItemText primary={<Typography variant={isMobile ? "h6" : "h5"} style={{ fontWeight: 'bold', color: 'white' }}> About</Typography>}/>    
        </ListItemButton>
      </List>
    </div>
  );
}

export default Sidebar;
