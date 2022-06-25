from flask import Flask
app = Flask('mle-airbus-detection')

@app.route('/')
def hello():
    return "Hello World!\n"

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8080)