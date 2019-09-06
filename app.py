import os
from flask import Flask, render_template, request, send_from_directory
import subprocess

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, "static/images/")
    # print(target)
    filename = ''

    if not os.path.isdir(target):
        os.mkdir(target)
    # endif
    for file in request.files.getlist("file"):
        # print(file.filename, 'file')
        filename = file.filename
        destination = "/".join([target, filename])
        # print(destination, 'destination')
        file.save(destination)
    # endfor
    # run ocr script
    cmd = 'python TrainAndTest.py '+ os.path.dirname(__file__)+"/static/images/"+filename
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    output = out.decode("utf-8") #output is bytes, so we decode it to string

    return render_template("result.html", textData=output, image_name=filename)

if __name__ == "__main__":
    app.run(port=4555, debug=True)
