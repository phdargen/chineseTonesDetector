import React from 'react';
import { List, ListItem, ListItemText } from '@mui/material';
import { Link } from 'react-router-dom';
import { Drawer, Box, Typography } from '@mui/material';

function Sidebar() {
  return (
    <div className="sidebar">
      <Typography variant="h4">Chinese Tones</Typography>

      <Box sx={{ padding: '16px' }}>
        <img 
          src="logo.png" 
          style={{ width: '100%', marginBottom: '16px' }}
        />
      </Box>

      <List>
        <ListItem button component={Link} to="/listening">
          <ListItemText primary="Listening" />
        </ListItem>
        <ListItem button component={Link} to="/speaking">
          <ListItemText primary="Speaking" />
        </ListItem>
        <ListItem button component={Link} to="/about">
          <ListItemText primary="About" />
        </ListItem>
      </List>
    </div>
  );
}

export default Sidebar;
