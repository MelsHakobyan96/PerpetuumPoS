from flask import Flask, render_template, request, redirect, url_for, session, g
from src.functions import *
import time
from werkzeug.serving import run_simple

app = Flask(__name__,  template_folder='./src/templates',
            static_folder='./src/static')

FILE_NAMES = None


@app.before_request
def before_request_func():
    app.jinja_env.cache = {}
    if request.method == 'POST':
        data = random_data()
        global FILE_NAMES
        FILE_NAMES = [data['episode_1_name'], data['episode_2_name']]
        convert_episodes(data)


@app.after_request
def after_request_func(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'GET':
        return render_template("index.html")

    elif request.method == 'POST':
        if FILE_NAMES is not None:
            value = submit(request.form)
            save_csv(FILE_NAMES + [value])

        time.sleep(15)

        return render_template("index.html")


if __name__ == "__main__":
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.jinja_env.auto_reload = True
    files = extra_files()

    run_simple('localhost', 5000, app,
               use_reloader=True, use_debugger=True)
