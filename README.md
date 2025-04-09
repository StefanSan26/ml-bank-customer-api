# Deployment of Machine Learning Project: Bank Customer Churn Prediction


### Option 1. Running Locally

```bash
docker build -t app-name .

docker run -p 80:80 app-name
```


### Option 2. Deployment With Heroku

```bash
heroku login
heroku create your-app-name
heroku git:remote your-app-name
heroku stack:set container
git push heroku main
```

### Use the API


```bash
curl -X POST https://bank-customer-churn-a4461a9d7ebe.herokuapp.com/predict \
-H "Content-Type: application/json" \
-d '{
    "CreditScore": 750,
    "Geography": "France",
    "Gender": "Female",
    "Age": 35,
    "Tenure": 5,
    "Balance": 125000.00,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 75000.00,
    "Surname": "Smith"
}'
```

Expected response:
```json
{
    "churn_probability": 0.1717,
    "will_churn": false
}
```

Note: The API validates the input data with the following constraints:
- CreditScore: Must be between 300 and 850
- Geography: Must be one of: France, Spain, Germany
- Age: Must be between 18 and 100
- HasCrCard and IsActiveMember: Must be 0 or 1
- All numeric fields must be positive values


