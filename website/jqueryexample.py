# -*- coding: utf-8 -*-
"""
ArtSift
"""
from pymongo import MongoClient
import json
from flask import Flask, jsonify, render_template, request
app = Flask(__name__)
client = MongoClient()
db = client.asi_database
c = db.asi_collection

@app.route('/_add_numbers')
def add_numbers():
    """Add two numbers server side, ridiculous but well..."""
    a = request.args.get('a', 0, type=int)
    b = request.args.get('b', 0, type=int)
    return jsonify(result=a + b)

@app.route('/_get_artist_count')
def get_artist_count():
	artistName = request.args.get('artistName', 0, type=str)
	#return jsonify(result=c.find({"artistName":artistName}).count())
	return jsonify(result=c.count())

@app.route('/_get_artist_data')
def get_artist_data():
	print "here"
	artistName = request.args.get('artistName', 0, type=str)
	#list(c.find({"artistName":artistName}, limit=20))
	return jsonify(result = list(c.find({"artistName":artistName}, limit=20)))
	
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run()
