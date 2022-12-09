from flask import Flask, render_template


def create_app():

    app = Flask(__name__)

    app.config['SECRET_KEY'] = 'dev'

    render_template("main.html")

    return app
