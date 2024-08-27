# FMCF: A Fusing Multiple Code Features Approach Based on Transformer for Solidity Smart Contracts Source Code Summarization
 In this paper, we propose a fusing multiple code
features (FMCF) approach based on Transformer for Solidity summarization.
First, FMCF created contract integrity modelling and state immutability
modelling in the data preprocessing stage to process and filter data that
meets security conditions. At the same time, FMCF retains the self-attention
mechanism to construct the Graph Attention Network (GAT) encoder and
CodeBERT encoder, which  respectively extract multiple feature vectors of
the code to ensure the integrity of the source code information. Furthermore,
the FMCF uses a weighted summation method to input these two types
of feature vectors into the feature fusion module for fusion and inputs the
fused feature vectors into the Transformer decoder to obtain the final smart
contract code summarization.
## Datasets and Model
The smart contract corpus we use comes from this website:
- https://zenodo.org/record/4587089#.YEMmWugzYuU
- https://github.com/NTDXYG/CCGIR
  
At the same time, in order to better demonstrate our method, we have saved the trained models and the organized Solidity dataset in Google Cloud Drive for everyone to download.
The checkpoints and dataset addresses are as follows:
- [Checkpoints and Datasets-Google in Cloud Drive](https://drive.google.com/drive/folders/1bHcTh2iOQOOhMDP_JwFwcxY56A2TX-rz?usp=drive_link)
## Data preprocessing
- The Solidity smart contract corpus code was converted to AST using the recognized Solidity Parser Antlr parsing package.https://github.com/federicobond/solidity-parser-antlr
- We use the Keras framework to establish a Transformer model. Keras is an open-source deep learning framework built on Python and provides a concise and efficient way to build and train deep neural networks. [The Keras official website](https://keras.io/) provides detailed documentation, API references, and sample code. You can learn about various features and usage of Keras here.
- Almost all of the code is built on the basis of Tensorflow, while referring to the official tutorial of [Tensorflow](https://www.tensorflow.org/tutorials/quickstart/advanced?hl=zh-cn), and we thank the contributors.
## CodeBERT Embedding
CodeBERT is a pre-trained model designed specifically for source code and natural language processing. It can understand the semantics of code and capture source code sequence information through a deep bidirectional Transformer architecture. This type of encoder not only understands individual syntax structures when processing source code text, but also grasps the overall program logic and functional intent.
- Installation dependencies: First, ensure that you have installed the necessary software and libraries, including Python, PyTorch, transformers, and tokenizers. You can use pip or conda for installation.
- Load pre trained model: Select a CodeBERT pre trained model suitable for your task from the [Hugging Face](https://huggingface.co/microsoft/codebert-base) model library, such as Microsoft/codebert base. Here, AutoModel and AutoToken are used to load the model and tokenizer.
- Alternatively, using the BERT client server mode, the pre trained model can be downloaded to the local machine.
```
pip install bert-serving-server  
pip install bert-serving-client
```
We use huggingface/transformers framework to train the model. You can use our model like the pre-trained Roberta base. Now, We give an example on how to load the model.
```Python
import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
model.to(device)
```
## Code Structure
```
FMCF-Master-Main
├─- README.md
├── Data_Preprocessing
│   ├── Deduplication.py
│   ├── merge_code.py
│   ├── contract_parse.py
│   ├── ast1.json
│   └── ...
├── Decoder
│   ├── DecoderModel.py
│   ├── Feature_Fusion.py
│   ├── Joint_Fusion_Decoder.py
│   └── Learning_Rate.py
├── Encoder
│   ├── Built_Transformer.py
│   ├── CodeBERT_Embedding.py
│   ├── CodeBERT_Encoder.py
│   ├── EncoderModel.py
│   ├── GAT_GraphEncoder.py
│   ├── GAT_Keras.py
│   └── Multi_Head_Attention.py
├── FMCF_Model
│   ├── Configs.py
│   ├── Evaluation.py
│   ├── EvaluationMetrics.py
│   ├── FMCF.py
│   └── Train.py
├── results
│   ├── FMCF_Results.txt
│   └── Parameter_ log.txt
├── DataOutput.py
└── init.py
```
## Requirements
```
Python 3.8
pytorch 1.7.0
Tensorflow 2.5.0
numpy 1.21.5
solidity-parser-antlr == 0.4.11
pickleshare 0.7.5
bert-serving-client 1.10.0  
bert-serving-server 1.10.0   
nltk 3.5
rough 1.0
```
## Acknowledgement
We borrowed and modified code from [NeuralCodeSum](https://github.com/wasiahmad/NeuralCodeSum), [CCGIR](https://github.com/NTDXYG/CCGIR). We would like to expresse our gratitdue for the authors of these repositeries.
