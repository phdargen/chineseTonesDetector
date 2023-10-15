
import { Box, Paper, Typography } from '@mui/material';

function Listening() {
  return (
    <div className="listening">
    <Typography variant="h1">Listen</Typography>
    <div>
      <Box mb={2}>
        <Paper elevation={3} style={{ padding: '16px' }}>
          <Typography variant="h6">Play sound</Typography>
        </Paper>
      </Box>

      <Box mb={2}>
        <Paper elevation={3} style={{ padding: '16px' }}>
          <Typography variant="h6">Select tone</Typography>
        </Paper>
      </Box>

      <Box mb={2}>
        <Paper elevation={3} style={{ padding: '16px' }}>
          <Typography variant="h6">Statistics</Typography>
        </Paper>
      </Box>

    </div>
    </div>

  );
}

export default Listening;
