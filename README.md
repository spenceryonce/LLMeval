[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/spenceryonce/LLMeval)
# LLM Evaluation
This library provides a collection of classes and functions to evaluate and compare different large language models (LLMs). The main purpose of the library is to build chatbots and evaluate their responses based on given objectives.

## Modules and Classes
1. **LanguageModelWrapper**
A base class for wrapping different language models.

2. **Prompt**
A class for managing prompt templates.

3. **BinaryPreference**
A class for managing binary preferences between two different responses.

4. **BinaryEvaluator**
A base class for evaluating binary preferences between two different responses.

5. **GPT35Evaluator**
A class for evaluating binary preferences using the GPT-3.5 LLM.

6. **OpenAIModel**
An enumeration class for listing available OpenAI LLM models.

7. **OpenAIGPTWrapper**
A class for wrapping OpenAI's GPT models.

8. **ClaudeWrapper**
A class for wrapping Anthropic's Claude LLM.

9. **CohereWrapper**
A class for wrapping Cohere's LLM.

10. **ChatBot**
A class for creating chatbot instances based on provided LLMs.

## Required Setup
1. Install dotenv (linux & mac), or python-dotenv (Windows)
```cmd
pip install python-dotenv
```
2. Install openai, cohere
```cmd
pip install openai cohere
```
OR
1. Install all from requirements.txt
```cmd
pip install -r requirements.txt
```

## Example Usage
### Import the required libraries
```python
import os
from dotenv import load_dotenv
import llm_eval

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
```
### We'll use GPT-3.5 as the evaluator.
```python
e = llm_eval.GPT35Evaluator(openai_api_key)
```
### Setup the Objective & Initial User Chats
```python
objective = "We're building a chatbot to discuss a user's travel preferences and provide advice."

# Chats that have been launched by users.
travel_chat_starts = [
    "I'm planning to visit Tulsa in spring.",
    "I'm looking for the cheapest flight to Spain today."
]
```
### Create the AI Models
```python
cohere_model = llm_eval.CohereWrapper(cohere_api_key)
davinici3_model = llm_eval.OpenAIGPTWrapper(openai_api_key, model=llm_eval.OpenAIModel.DAVINCI3.value)
chatgpt35_model = llm_eval.OpenAIGPTWrapper(openai_api_key)
```
### Run The Evaluator for Each User Chat
```python
for tcs in travel_chat_starts:

    messages = [{"role":"system", "content":objective},
            {"role":"user", "content":tcs}]

    response_cohere = cohere_model.complete_chat(messages, "assistant")
    response_gpt35 = chatgpt35_model.complete_chat(messages, "assistant")

    response_davinvi3 = davinici3_model.complete_chat(messages, "assistant")

    pref = e.choose(objective, tcs, response_cohere, response_gpt35)
    print(f"1: {response_cohere}")
    print(f"2: {response_gpt35}")
    print(f"Preferred Choice: {pref}")

    pref2 = e.choose(objective, tcs, response_gpt35, response_davinvi3)
    print(f"1: {response_gpt35}")
    print(f"2: {response_davinvi3}")
    print(f"Preferred Choice: {pref2}")
```
