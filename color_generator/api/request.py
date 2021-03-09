import requests

url = 'http://localhost:8000/predict'
r = requests.post(url, data={'description': 'ferrari red'})


print(r)