import requests
import json

url = "http://localhost:5000/predictions"
data = json.dumps(
    {
        "input": {
            "prompt": "I'm an athletic and 171cm tall woman in my mid twenties, \
                I have a rectangle shaped body with slightly broad shoulders and have a sleek,\
                casual style. I usually prefer darker colors.",
            "event": "business meeting",
        }
    }
)

response = requests.post(url, data=data)
print(response.json())
