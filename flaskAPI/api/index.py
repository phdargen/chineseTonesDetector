from flask import Flask
from flask_cors import CORS  

app = Flask(__name__)
CORS(app, resources={
    r"/about": {"origins": ["http://localhost:3000/*", "https://chinese-tones-detector.vercel.app/*"]}
})


@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/about')
def about():
    return 'About'
