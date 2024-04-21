from  transformers  import pipeline

pipe = pipeline("sentiment-analysis")
pipe.device='mps'
# tasks are ['audio-classification', 'automatic-speech-recognition', 'conversational', 'depth-estimation', 'document-question-answering', 'feature-extraction', 'fill-mask', 'image-classification', 'image-segmentation', 'image-to-image', 'image-to-text', 'mask-generation', 'ner', 'object-detection', 'question-answering', 'sentiment-analysis', 'summarization', 'table-question-answering', 'text-classification', 'text-generation', 'text-to-audio', 'text-to-speech', 'text2text-generation', 'token-classification', 'translation', 'video-classification', 'visual-question-answering', 'vqa', 'zero-shot-audio-classification', 'zero-shot-classification', 'zero-shot-image-classification', 'zero-shot-object-detection', 'translation_XX_to_YY']"

question_answerer = pipeline(task="question-answering")
preds = question_answerer(
    question="what is the name of the repository?",
    context="The name of the repository is huggingface/transformers",
)
print(preds)

from transformers import TrainingArguments

model_dir = "models/bert-base-cased-finetune-yelp"

# logging_steps 默认值为500，根据我们的训练数据和步长，将其设置为100
training_args = TrainingArguments(output_dir=model_dir,
                                  per_device_train_batch_size=16,
                                  num_train_epochs=5,
                                  logging_steps=100,
                                  use_cpu=False,
                                  use_mps_device=True)

import numpy as np
import evaluate
