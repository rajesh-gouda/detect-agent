from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from starlette.responses import RedirectResponse
from urllib.parse import urlencode
from enum import Enum
from pymongo import MongoClient
from dataclasses import dataclass
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os
import logging
import sys
import uvicorn
from datetime import datetime
import requests
from openai import AsyncOpenAI
import json

# pyright: ignore[reportMissingImports]


def setup_logger(name="detect_agent") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    for handler in [
        logging.FileHandler("detect_agent.log"),
        logging.StreamHandler(sys.stdout),
    ]:
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


load_dotenv()
app = FastAPI()


BEARER_TOKEN = os.getenv("BEARER_TOKEN")
TWITTER_API_URL = "https://api.twitter.com/2/tweets"

logger = setup_logger()

logger.info("Server started")


class AgentType(str, Enum):
    investor = "investor"
    actor = "actor"
    actress = "actress"
    director = "director"
    script_writter = "script_writter"
    producer = "producer"
    technical_supporter_specialist = "technical_supporter_specialist"
    marketing_and_promotion_specialist = "marketing_and_promotion_specialist"
    casting_assistant = "casting_assistant"


AI_AGENT_DESCRIPTIONS = {
    AgentType.investor: (
        "An investor is a person or entity that provides financial support to a project, typically in exchange for equity or a share of the profits. "
        "They are crucial in funding film projects and ensuring their financial viability."
    ),
    AgentType.actor: (
        "An actor is a person who portrays a character in a film, television show, or theater production. "
        "They bring scripts to life through their performances and are essential for storytelling."
    ),
    AgentType.actress: (
        "An actress is a female performer who plays characters in films, television, or stage productions. "
        "She contributes to the narrative through expressive acting and emotional delivery."
    ),
    AgentType.director: (
        "A director is responsible for the creative vision of a film. "
        "They guide the cast and crew, make key decisions about style and pacing, and ensure the story is effectively told."
    ),
    AgentType.script_writter: (
        "A script writer creates the screenplay or dialogue for a film or show. "
        "They shape the narrative structure and character development, laying the foundation for the production."
    ),
    AgentType.producer: (
        "A producer oversees the production process, managing budgets, schedules, and coordination among departments. "
        "They ensure the film is completed on time and within budget."
    ),
    AgentType.technical_supporter_specialist: (
        "A technical supporter specialist handles the technical infrastructure, including lighting, sound, camera equipment, and software tools used in production."
    ),
    AgentType.marketing_and_promotion_specialist: (
        "A marketing and promotion specialist is responsible for creating awareness and excitement around the film. "
        "They manage campaigns, public relations, and promotional events."
    ),
    AgentType.casting_assistant: (
        "A casting assistant helps identify and audition suitable actors for various roles. "
        "They assist casting directors in coordinating casting calls and managing actor submissions."
    ),
}


@dataclass
class MongoConfig:
    host: str = os.getenv("HOST")  # host should be set in .env file
    port: int = 8002
    username: str = "root"
    password: str = os.getenv("PASSWORD")  # password should be set in .env file
    auth_db: str = "admin"


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Only for dev. Restrict in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


config = MongoConfig()
client = AsyncIOMotorClient(
    host=config.host,
    port=config.port,
    username=config.username,
    password=config.password,
    authSource=config.auth_db,
)
db = client["fuss_agent"]
collection = db["tweets"]

openai_client = AsyncOpenAI()


async def get_tweet(tweet_id):
    url = f"{TWITTER_API_URL}/{tweet_id}"
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
    }
    params = {"tweet.fields": "created_at,author_id,text,public_metrics"}
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        return response.json()["data"]["text"]
    else:
        print("Error:", response.status_code, response.text)
        return None


async def add_records(twit_id: str, agent_type: str):
    try:
        # Insert a sample document (creates DB and collection if not exist)
        doc = {
            "tweetId": twit_id,
            "agentType": agent_type,
            "createdAt": datetime.now(),
            "status": "pending",
            "updatedAt": datetime.now(),
        }
        print(f"Inserting document: {doc}")
        result = await collection.insert_one(doc)
        logger.info(f"Inserted with _id:{result.inserted_id}")
    except Exception as e:
        logger.error(f"Error inserting document: {e}")


async def get_agent_required(text: str):
    prompt = f"""
    You are an AI Agent Type Evaluator.
     
    Your task is to:
    1. Evaluate the customer's reply.
    2. Determine whether a response is needed.
    3. If a response is needed, identify the appropriate AI agent type based on the context.

    Here are the agent types with their descriptions:

    {chr(10).join([f"- {agent.value}: {desc}" for agent, desc in AI_AGENT_DESCRIPTIONS.items()])}

    Now analyze the following customer reply:
    "{text}"

    Output strictly in the following JSON format:

    {{
        "response_needed": true or false,  # True if the reply deserves a response, False otherwise
        "agent_type": "<agent_type>"       # One of: {', '.join([agent.value for agent in AgentType])}
    }}
    """
    logger.info(f"Prompt for OpenAI: {prompt}")
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content.strip()
        logger.info(f"OpenAI raw response: {content}")
        result = json.loads(content)
        response_needed = result.get("response_needed", False)
        agent_type = result.get("agent_type", "").lower()

        logger.info(f"Response needed: {response_needed}")
        logger.info(f"Agent type determined: {agent_type}")
        return {"response_needed": response_needed, "agent_type": agent_type}
    except Exception as e:
        logger.error(f"Error determining agent type: {e}")
        raise HTTPException(
            status_code=500, detail="Unable to process the request please try again."
        )


@app.post("/detect-agent")
async def detect_agent(twitter_id: str = Form(...)):
    logger.info(f"Received Twitter IDs: {twitter_id}")
    if not twitter_id:
        return JSONResponse(
            status_code=400,
            content={"message": "Twitter ID is required."},
        )
    try:
        tweet_data = await get_tweet(twitter_id)
        # tweet_data = "you did great job in the movie as a actor, I loved it"
        if tweet_data:
            res = await get_agent_required(tweet_data)
            if res["response_needed"]:
                agent_type = res["agent_type"]
                if not agent_type:
                    return JSONResponse(
                        status_code=400,
                        content={"message": "No agent type determined."},
                    )
                await add_records(twitter_id, agent_type)
            else:
                logger.info("No response needed for this tweet.")
                return JSONResponse(
                    status_code=200,
                    content={"message": "No response needed for this tweet."},
                )
            return JSONResponse(
                status_code=200,
                content={"message": "Tweet data retrieved and record added."},
            )
        else:
            return JSONResponse(
                status_code=404,
                content={"message": "Tweet not found."},
            )
    except Exception as e:
        logger.error(f"Error retrieving tweet: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": "Cannot process the request at this time."},
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
