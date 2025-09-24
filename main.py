import os
import sys
import itertools
import llm_eval
from dotenv import load_dotenv


def get_api_keys():
    """Load and validate API keys from environment variables."""
    load_dotenv()

    possible_keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "cohere": os.getenv("COHERE_API_KEY"),
        "grok": os.getenv("GROK_API_KEY"),
        "mistral": os.getenv("MISTRAL_API_KEY"),
        "deepseek": os.getenv("DEEPSEEK_API_KEY"),
        "deepinfra": os.getenv("DEEPINFRA_API_KEY"),
    }

    available_keys = {k: v for k, v in possible_keys.items() if v}

    if len(available_keys) < 2:
        print("Error: At least two API keys must be set to run the evaluation.")
        sys.exit(1)

    return available_keys


def initialize_models(api_keys):
    """Initialize the language models for which API keys are available."""
    models = {}
    if "cohere" in api_keys:
        models["cohere"] = llm_eval.CohereWrapper(api_keys["cohere"])
    if "openai" in api_keys:
        models["davinci3"] = llm_eval.OpenAIGPTWrapper(
            api_keys["openai"], model=llm_eval.OpenAIModel.DAVINCI3.value
        )
        models["gpt3.5"] = llm_eval.OpenAIGPTWrapper(api_keys["openai"])
    if "grok" in api_keys:
        models["grok"] = llm_eval.GrokWrapper(api_keys["grok"])
    if "mistral" in api_keys:
        models["mistral"] = llm_eval.MistralWrapper(api_keys["mistral"])
    if "deepseek" in api_keys:
        models["deepseek"] = llm_eval.DeepSeekWrapper(api_keys["deepseek"])
    if "deepinfra" in api_keys:
        models["llama3"] = llm_eval.Llama3Wrapper(api_keys["deepinfra"])

    return models


def run_evaluation(evaluator, objective, chat_start, model1_name, model1, model2_name, model2):
    """Run a single evaluation between two models."""
    messages = [
        {"role": "system", "content": objective},
        {"role": "user", "content": chat_start},
    ]

    response1 = model1.complete_chat(messages, "assistant")
    response2 = model2.complete_chat(messages, "assistant")

    preference = evaluator.choose(objective, chat_start, response1, response2)

    print(f"Comparing {model1_name} and {model2_name}:")
    print(f"1: {response1}")
    print(f"2: {response2}")
    print(f"Preferred Choice: {preference}\n")


def main():
    """Main function to run the LLM evaluation."""
    api_keys = get_api_keys()
    models = initialize_models(api_keys)

    # Use the first available model as the evaluator
    evaluator_model_name = sorted(models.keys())[0]
    evaluator = llm_eval.LLMEvaluator(models[evaluator_model_name])
    print(f"Using {evaluator_model_name} as the evaluator.\n")

    objective = "We're building a chatbot to discuss a user's travel preferences and provide advice."
    travel_chat_starts = [
        "I'm planning to visit Tulsa in spring.",
        "I'm looking for the cheapest flight to Spain today.",
    ]

    model_pairs = list(itertools.combinations(models.items(), 2))

    for chat_start in travel_chat_starts:
        print(f"--- Evaluating for chat start: '{chat_start}' ---")
        for (model1_name, model1), (model2_name, model2) in model_pairs:
            run_evaluation(
                evaluator, objective, chat_start, model1_name, model1, model2_name, model2
            )


if __name__ == "__main__":
    main()