import os
from dotenv import load_dotenv
import llm_eval

load_dotenv()
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY cannot be null or empty")
e = llm_eval.GPT35Evaluator(openai_api_key)

if not anthropic_api_key:
    raise ValueError("ANTHROPIC_API_KEY cannot be null or empty")

if not cohere_api_key:
    raise ValueError("COHERE_API_KEY cannot be null or empty")
# We'll use GPT-3.5 as the evaluator.
e = llm_eval.GPT35Evaluator(openai_api_key)

objective = "We're building a chatbot to discuss a user's travel preferences and provide advice."

# Chats that have been launched by users.
travel_chat_starts = [
    "I'm planning to visit Tulsa in spring.",
    "I'm looking for the cheapest flight to Spain today."
]

cohere_model = llm_eval.CohereWrapper(cohere_api_key)
davinici3_model = llm_eval.OpenAIGPTWrapper(openai_api_key, model=llm_eval.OpenAIModel.DAVINCI3.value)
chatgpt35_model = llm_eval.OpenAIGPTWrapper(openai_api_key)

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