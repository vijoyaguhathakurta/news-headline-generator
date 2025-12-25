# First install flask -> pip install flask
from flask import Flask,render_template,request 
# From module flask import class Flask,Need render_template() to render HTML pages
from text_summary import summarize#importing our backend pythonfile

app = Flask(__name__) # Construct an instance of Flask class for our webapp

#routing  
@app.route('/')# URL '/' to be handled by main() route handler
def index():
    #Render an HTML template and return
    return render_template('index.html')

#Passing the raw text (which is recieved from the index.htmlpage) 
# to get the summarize text showing the result redirecting to another html page
@app.route('/analyze',methods=['GET','POST'])
def analyze():
    if request.method =='POST':
        rawtext=request.form['rawtext']
        summary=summarize(rawtext)
    return render_template('summary.html',orignal_txt = rawtext,summary = summary)
    # HTML file to be placed under sub-directory templates

if __name__=="__main__":# Script executed directly?
    app.run(debug=True,port=5000)# Launch built-in web server and run this Flask webapp

