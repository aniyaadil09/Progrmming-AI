from flask import Flask, render_template
import requests

app = Flask(__name__)

# Serve the frontend page
@app.route("/")
def home():
    return render_template("index.html")

# API route to fetch random joke
@app.route("/joke")
def joke():
    try:
        url = "https://official-joke-api.appspot.com/random_joke"
        response = requests.get(url)
        data = response.json()
        return f"<b>{data['setup']}</b><br><br>{data['punchline']}"
    except:
        return "Sorry, could not fetch a joke right now."

if __name__ == "__main__":
    app.run(debug=True)