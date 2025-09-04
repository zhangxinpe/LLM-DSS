# 




# LLM-DSS

Code for our paper:
MLLM-based dual-stream synergistic model for multimodal aspect-based sentiment analysis

## Abstract

![模型图](./main.png)Multimodal aspect-based sentiment analysis aims to determine sentiment polarity toward specific aspects within image–text content, supporting public opinion analysis and information mining. Existing methods face two main challenges: (i) cross-modal interaction can impede the learning of syntactic structures; (ii) external knowledge is underutilized for enriching aspect context. To address these challenges, we propose MLLM-based dual-stream synergistic model (LLM-DSS). Our model first leverages a multimodal large language model to generate external knowledge and image descriptions, and subsequently conducts synergistic processing through a parallel dual-stream architecture. Specifically, the knowledge-enhanced multimodal stream fuses images, text, and external knowledge via knowledgeenhanced attention, while the syntax-enhanced text stream parses syntactic structure with syntax-enhanced attention. Finally, a synergistic alignment module enables information transfer between the two streams. Experiments on two public MABSA datasets indicate that our method achieves state-of-the-art performance.



## Prompt Template

![提示词模板](./prompt.png)

### Main Results 

![Result](./result.png)


## Citation

If our work has been helpful to you, please mark references to our work in your research and thank you for your support.

## Note

* Code of this repo heavily relies on [IOT](https://github.com/qlwang25/image2text_conversion).
