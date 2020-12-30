from flask import Flask, render_template, request, redirect, url_for
from funcs.functions import *
# import .utils

app = Flask(__name__)

FILE_NAMES = None


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'GET':
        data = random_data()
        global FILE_NAMES
        FILE_NAMES = [data['episode_1_name'], data['episode_2_name']]

        convert_episodes(data)
        return render_template("index.html")

    elif request.method == 'POST':
        if FILE_NAMES is not None:
            value = submit(request.form)
            save_csv(FILE_NAMES + [value])

        return redirect('/')


if __name__ == "__main__":
    app.run(debug=True)
