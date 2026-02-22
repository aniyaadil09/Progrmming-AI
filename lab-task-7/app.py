from flask import Flask
import requests

app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to My Joke App!"

@app.route("/joke")
def joke():
    url = "https://official-joke-api.appspot.com/random_joke"
    response = requests.get(url)
    data = response.json()
    return f"{data['setup']} <br><br> {data['punchline']}"

if __name__ == "__main__":
    app.run(debug=True)