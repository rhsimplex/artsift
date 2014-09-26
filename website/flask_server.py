from flask import Flask, session, redirect, url_for, escape, request, Response, jsonify
from functools import wraps
from pymongo import MongoClient

app = Flask(__name__)
client = MongoClient()
db = client.asi_database
c = db.asi_collection
USERNAME = 'admin'
PASSWORD = 'default'
SECRET_KEY = 'development key'

@app.route("/")
def index():
    return render_template('index.html')

def get_artist_entries():
    name = request.args.get('artistName', 0, type=str)
    return jsonify(result=c.find({"artistName":name})

if __name__ == "__main__":
    app.run()
