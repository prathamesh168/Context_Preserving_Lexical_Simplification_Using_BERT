# from crypt import methods
# from urllib import request
from flask import Flask,redirect,url_for,render_template,request

# { % ... % } for statements
# { { } } expressions to print output
# { # .... # } this is for internal comment
from CWI import *
###  creates WSGI app
app = Flask(__name__)
#decorator url
@app.route('/')
def welcome():
    return "HELLO"

@app.route('/maths/<int:marks>')
def maths(marks):
    if marks<50:
        result = "fail"
    else:
        result="pass1"
    return redirect(url_for(result,score=marks))

@app.route('/pass1/<int:score>')
def pass1(score):
    return ("you are passed and your marks are " + str(score))

@app.route('/fail/<int:score>')
def fail(score):
    return ("you are failed and your marks are" + str(score))


@app.route('/submit')
def submit():
    return render_template('index.html')

@app.route('/result',methods=['POST','GET'])
def result():
    
    complex1 =" "
    if request.method == 'POST':
        sentence = request.form['sentence']
        complex_word = request.form['complex_word']
        complex1 = CWI(sentence,model,clf_model12)
    return(render_template('results.html',complex=complex1))


if __name__ == '__main__':
    app.run(debug=True)