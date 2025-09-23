import os
import sys
import llm_eval
from dotenv import load_dotenv


def get_api_keys():
    """Load and validate API keys from environment variables."""
    load_dotenv()
    api_keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "cohere": os.getenv("COHERE_API_KEY"),
    }

    if not api_keys["openai"]:
        print("Error: OPENAI_API_KEY not found in .env file or environment variables.")
        sys.exit(1)
    if not api_keys["cohere"]:
        print("Error: COHERE_API_KEY not found in .env file or environment variables.")
        sys.exit(1)

    return api_keys


def initialize_models(api_keys):
    """Initialize the language models."""
    return {
        "cohere": llm_eval.CohereWrapper(api_keys["cohere"]),
        "davinci3": llm_eval.OpenAIGPTWrapper(
            api_keys["openai"], model=llm_eval.OpenAIModel.DAVINCI3.value
        ),
        "gpt3.5": llm_eval.OpenAIGPTWrapper(api_keys["openai"]),
    }


def run_evaluation(evaluator, objective, chat_start, model1, model2):
    """Run a single evaluation between two models."""
    messages = [
        {"role": "system", "content": objective},
        {"role": "user", "content": chat_start},
    ]

    response1 = model1.complete_chat(messages, "assistant")
    response2 = model2.complete_chat(messages, "assistant")

    preference = evaluator.choose(objective, chat_start, response1, response2)

    print(f"Comparing {model1.__class__.__name__} and {model2.__class__.__name__}:")
    print(f"1: {response1}")
    print(f"2: {response2}")
    print(f"Preferred Choice: {preference}\n")


def main():
    """Main function to run the LLM evaluation."""
    api_keys = get_api_keys()

    # We'll use GPT-3.5 as the evaluator.
    evaluator = llm_eval.GPT35Evaluator(api_keys["openai"])
    models = initialize_models(api_keys)

    objective = "We're building a chatbot to discuss a user's travel preferences and provide advice."
    travel_chat_starts = [
        "I'm planning to visit Tulsa in spring.",
        "I'm looking for the cheapest flight to Spain today.",
    ]

    for chat_start in travel_chat_starts:
        print(f"--- Evaluating for chat start: '{chat_start}' ---")
        run_evaluation(
            evaluator, objective, chat_start, models["cohere"], models["gpt3.5"]
        )
        run_evaluation(
            evaluator, objective, chat_start, models["gpt3.5"], models["davinci3"]
        )


if __name__ == "__main__":
    main()