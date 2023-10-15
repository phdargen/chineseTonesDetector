import React from 'react';
import './App.css';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Sidebar from './Sidebar';
import Listening from './Listening';
import About from './About';
import VoiceRecordingButton from './VoiceRecordingButton';

import { ThemeProvider, CssBaseline } from '@mui/material';
import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    primary: {
      light: '#757ce8',
      main: '#3f50b5',
      dark: '#002884',
      contrastText: '#fff',
    },
    secondary: {
      light: '#ff7961',
      main: '#f44336',
      dark: '#ba000d',
      contrastText: '#000',
    },
  },
});
// function App() {
//   return (
//     <div className="App">
//       <h1>Check my tones</h1>
//       <VoiceRecordingButton />
//     </div>
//   );
// }

function App() {
  return (
    <ThemeProvider theme={theme}>
        <CssBaseline />
        <Router>
          <div className="app-container">
            <Sidebar />
            <div className="content">
              <Routes>
                <Route path="/listening" element={ <Listening/>} />
                <Route path="/speaking" element={ <VoiceRecordingButton/>} />
                <Route path="/about" element={ <About/>} />
                <Route path="/" exact element={<Listening/>} /> {/* Default route */}
              </Routes>
            </div>
          </div>
        </Router>
    </ThemeProvider>
  );
}

export default App;
