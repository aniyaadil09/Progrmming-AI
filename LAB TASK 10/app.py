from flask import Flask, render_template, request

app = Flask(__name__)

# chatbot logic
def chatbot_response(user_input):
    user_input = user_input.lower()

    if "admission" in user_input:
        return "Admissions are open. Last date is 30 August."
    
    elif "fee" in user_input:
        return "Fee is approximately 50,000 per semester."
    
    elif "programs" in user_input:
        return "We offer BSCS, BBA, BSIT and more."
    
    elif "deadline" in user_input:
        return "Deadline is 30 August."
    
    elif "hello" in user_input or "hi" in user_input:
        return "Hello! How can I help you?"
    
    else:
        return "Sorry, I don't understand."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_response():
    user_text = request.form["msg"]
    return chatbot_response(user_text)

if __name__ == "__main__":
    app.run(debug=True)