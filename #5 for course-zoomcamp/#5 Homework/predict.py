from flask import Flask
from flask import request
from flask import jsonify
from flask import Flask

import pickle



model_file = 'model_C=1.0.bin'


app = Flask('credit card')

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

@app.route('/predict', methods=['POST'])
def predict():


    client = request.get_json()

    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0,1]
    churn = y_pred >= 0.5


    result = {
        'churn_probability':float(y_pred)

    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')




