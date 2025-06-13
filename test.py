from flask import Flask

app = Flask(__name__)

print("Flask app instance created.")

if __name__ == "__main__":
    print("Starting Flask server on http://0.0.0.0:8080")
    app.run(host="0.0.0.0", port=8080)