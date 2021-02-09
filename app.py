from flask import Flask,render_template,request
import pickle as pkl
import numpy as np

model = pkl.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/', methods = ['POST','GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def prediction():

        data1 = request.form.get("field1",False)
        data2 = request.form.get("field2",False)
        data3 = request.form.get("field3",False)
        data4 = request.form.get("field4",False)
        input_data = np.array([[data1,data2,data3,data4]]) 
        prediction = model.predict(input_data)
        return render_template('main.html', data = prediction)

if __name__ == '__main__':
    app.run(host= 'localhost' , port= 5000, debug= True)


