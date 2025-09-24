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

10. **GrokWrapper**
A class for wrapping Grok's models.

11. **MistralWrapper**
A class for wrapping Mistral's models.

12. **DeepSeekWrapper**
A class for wrapping DeepSeek's models.

13. **Llama3Wrapper**
A class for wrapping Llama 3 models via DeepInfra.

14. **ChatBot**
A class for creating chatbot instances based on provided LLMs.

## Required Setup
1. Install all from requirements.txt
```cmd
pip install -r requirements.txt
```
2. Create a `.env` file in the root of the project and add the following API keys:
```
OPENAI_API_KEY=your_openai_api_key
COHERE_API_KEY=your_cohere_api_key
GROK_API_KEY=your_grok_api_key
MISTRAL_API_KEY=your_mistral_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPINFRA_API_KEY=your_deepinfra_api_key
```

## Example Usage
The `main.py` script provides an example of how to use the library. It initializes all the supported models, defines an objective, and then runs a series of evaluations comparing each model to GPT-3.5.

To run the example:
```bash
python main.py
```
