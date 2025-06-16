# detect-twitter-agent


## Quick Start
1. clone this repo
2. create a .env file with with mongo HOST, PASSWORD, OPENAI_API_KEY, BEARER_TOKEN(twitter api key)
3. create docker with command `docker build -t detect-agent .`
4. run docker with `docker run docker run -d --name detect-agent -p 5007:5007 detect-agent:latest`