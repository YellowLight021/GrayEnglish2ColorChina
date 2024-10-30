---
language: 
- en 
- zh 
tags:
- translation
widget: 
- text: "I like to study Data Science and Machine Learning."
---

# liam168/trans-opus-mt-en-zh

## Model description

* source group: English
* target group: Chinese
* model: transformer
* source language(s): eng
* target language(s): cjy_Hans cjy_Hant cmn cmn_Hans cmn_Hant gan lzh lzh_Hans nan wuu yue yue_Hans yue_Hant

## How to use

```python
>>> from transformers import AutoModelWithLMHead,AutoTokenizer,pipeline
>>> mode_name = 'liam168/trans-opus-mt-en-zh'
>>> model = AutoModelWithLMHead.from_pretrained(mode_name)
>>> tokenizer = AutoTokenizer.from_pretrained(mode_name)
>>> translation = pipeline("translation_en_to_zh", model=model, tokenizer=tokenizer)
>>> translation('I like to study Data Science and Machine Learning.', max_length=400)
    [{'translation_text': '我喜欢学习数据科学和机器学习'}]
```

## Contact

liam168520@gmail.com
