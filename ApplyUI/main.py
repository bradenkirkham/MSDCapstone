import pymongo
from flask import Flask, render_template, request, flash, redirect
import os
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from scraper import glass_scraper
import utils

UPLOAD_FOLDER = '/Users/bkirkham/FinalProject/ApplyUI/Uploads'
ALLOWED_EXTENSIONS = {'doc', 'docx'}

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def create_app():
    app.config['SECRET_KEY'] = 'dev'

    return render_template("main.html", visibility = "hidden", job_data = [1,2,3,4,5,6,7,8,9,10])


# this probably needs to be renamed process data.
@app.route("/upload_data", methods=['POST'])
def upload_data():

    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    client = MongoClient("")

    db = client.ApplyAI

    resumes = db.Resumes

    job_data_clean = db.JobDataClean

    job_data_raw = db.JobDataRaw

    corpus_idf = db.CorpusIDF

    if 'user-file' not in request.files:
        flash('No file part')
        return redirect('/')

    resume = request.files['user-file']

    if resume.filename == '':
        flash('No file selected')
        return redirect('/')

    if resume:
        filename = secure_filename(resume.filename)
        post = {"_id": filename, "resume": resume.read()}
        resumes.insert_one(post)

    job_post = request.form.get("job-URL")

    job_post = glass_scraper(job_post)
    job_data_raw.insert_one(job_post)

    job_post = utils.data_cleaner(job_post)
    job_data_clean.insert_one(job_post)

    job_post_tf = utils.tf_calculator(job_post['Description'])

    job_corpus_idf = corpus_idf.find_one({}, {'_id': 0})

    job_post_tfidf = utils.tf_idf(job_corpus_idf, job_post_tf)

    print(job_post_tfidf[0:10])
    # dynamically update html revealing hidden divs and adding to the divs the list of important terms.
    # recalculate tf-idf and update database.

    corpus_idf.delete_one(job_corpus_idf)

    all_clean_data = job_data_clean.find({}, {'_id': 0, 'Company': 0})

    corpus = []

    for data in all_clean_data:
        Description = data['Description']
        Description = "".join(word + " " for word in Description)
        corpus.append(Description)

    updated_idf = utils.idf_calculator(corpus)

    corpus_idf.insert_one(updated_idf)

    # i currently have it redirecting to home, i think i can't do that anymore
    # I'm pretty sure there will be no return/redirect and it will just dynamically update with the list.
    return render_template('main.html', visibility="visible", job_data = job_post_tfidf)


if __name__ == '__main__':
    app.run(debug=True)
