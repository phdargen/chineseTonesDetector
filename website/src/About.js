import React, { useState, useEffect } from 'react';

const About = () => {
  const [aboutText, setAboutText] = useState('');

  useEffect(() => {
    fetch('https://flask-test-nu-one.vercel.app/about')  
      .then(response => response.text())
      .then(text => setAboutText(text))
      .catch(err => console.error('Failed to fetch About data:', err));
  }, []);

  return <h1>{aboutText || 'Loading...'}</h1>;
};

export default About;
