from flask import Flask, render_template, request, redirect, url_for, session, g
from funcs.functions import *
import time

app = Flask(__name__)

FILE_NAMES = None


@app.after_request
def after_request_func(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers['Cache-Control'] = 'public, max-age=0'
    response.headers.add('Accept-Ranges', 'bytes')


    if '302' in response.status:
        time.sleep(30)
        print(response)
        return redirect(url_for('index'))
    return response



@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'GET':
        data = random_data()
        global FILE_NAMES
        FILE_NAMES = [data['episode_1_name'], data['episode_2_name']]
        print(FILE_NAMES)

        convert_episodes(data)
        return render_template("index.html")

    elif request.method == 'POST':
        if FILE_NAMES is not None:
            value = submit(request.form)
            save_csv(FILE_NAMES + [value])

        return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(debug=True)
