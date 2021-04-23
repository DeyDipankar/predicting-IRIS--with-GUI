from flask import Flask,render_template,request
import pickle as pkl
import numpy as np

model = pkl.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/', methods = ['POST','GET'])
def home():
    return render_template('main.html')

@app.route('/predict', methods=['POST', 'GET'])
def prediction():

        data1 = request.form.get("field1",False)
        data2 = request.form.get("field2",False)
        data3 = request.form.get("field3",False)
        data4 = request.form.get("field4",False)
        input_data = np.array([[data1,data2,data3,data4]]) 
        prediction = model.predict(input_data)
        #if prediction
        if prediction == 1:
            data = "It's a Setosa"
            path = '/static/setosa.jpg'
        elif prediction == 0:
            data = "It's a Versicolor"
            path = '/static/versicolor.jpg'
        else:
            data = "It's a Virginica"
            path = '/static/virginica.jpg'

        return render_template('main.html', data = data, image_file = path)

if __name__ == '__main__':
    app.run(host= 'localhost' , port= 5000, debug= True)



