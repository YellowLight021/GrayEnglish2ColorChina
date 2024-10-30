from optimum.intel.openvino import OVModelForSeq2SeqLM
from transformers import AutoTokenizer


class Translation():
    def __init__(self,model_dir,model_name="entoch_model"):
        self.model = OVModelForSeq2SeqLM.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def from_en_to_ch(self, text):
        # 加载预训练模型
        # 读取文本2
        tokenized_text = self.tokenizer([text], return_tensors="pt")
        #
        # 得到预测出的token
        translation = self.model.generate(
            **tokenized_text)  # 执行翻译，返回翻译后的tensor
        #
        # 将预测出的token转为单词
        translated_text = self.tokenizer.batch_decode(
            translation, skip_special_tokens=True)
        return translated_text