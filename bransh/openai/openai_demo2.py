import openai
import os

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

# Replace 'your-api-key' with your actual OpenAI API key
openai.api_key = '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'

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Which vitamin is supplied from only animal source:A.Vitamin C B. Vitamin B7 C.Vitamin B12 D. Vitamin D"},
    ]
)

print(response['choices'][0]['message']['content'])
