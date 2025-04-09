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
