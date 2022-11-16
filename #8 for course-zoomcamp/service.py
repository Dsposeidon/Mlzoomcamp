from re import M
from unittest import result
import bentoml
from bentoml.io import JSON

model_ref = bentoml.xgboost.get("credit_risk_model:latest")
dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()

service = bentoml.Service("credit_risk_classifier", runners=[model_runner])

@service.api(input=JSON(), output=JSON())
def classify(app_data):
    vector = dv.transform(app_data)
    prediction = model_runner.predict.run(vector)
    print(prediction)

    result = prediction[0]

    if result > 0.5 :
        return{"status":'Declined'}
    elif result > 0.23:
        return{"status":" Good enough"}
    else:
        return {"status":"Approved"}
    
    