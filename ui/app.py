from flask import Flask, render_template, request
from funcs.functions import *
# import .utils

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'GET':
        return render_template("index.html")

    elif request.method == 'POST':
        value = submit(request.form)
        print(value)
        return render_template("index.html")




if __name__ == "__main__":
    app.run(debug=True)
