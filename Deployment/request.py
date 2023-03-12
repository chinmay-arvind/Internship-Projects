# Source code adapted from: https://towardsdatascience.com/deploy-a-machine-learning-model-using-flask-da580f84e60c
# Importing necessary libraries
import requests

# Requests the server for predictions
website = 'http://localhost:5000/api'
new_request = requests.post(website,json={'exp':1.8,})
print(new_request.json())