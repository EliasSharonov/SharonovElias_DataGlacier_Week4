from flask import Flask, request

import joblib
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()

model = joblib.load('iris_model.pkl')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        prediction = model.predict(np.array([[sepal_length, sepal_width, petal_length, petal_width]]))

        species = iris.target_names[prediction[0]]

        return f'<h1>Predicted Iris Species: {species}</h1>'

    return '''
        <form method="post">
            <Title> Iris Classifier </Title>
            <h1> Iris Classifier </h1>
            <label for="sepal_length">Sepal Length:</label>
            <input type="text" id="sepal_length" name="sepal_length"><br><br>
            <label for="sepal_width">Sepal Width:</label>
            <input type="text" id="sepal_width" name="sepal_width"><br><br>
            <label for="petal_length">Petal Length:</label>
            <input type="text" id="petal_length" name="petal_length"><br><br>
            <label for="petal_width">Petal Width:</label>
            <input type="text" id="petal_width" name="petal_width"><br><br>
            <input type="submit" value="Run the Model">
        </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)