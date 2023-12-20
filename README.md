# Perception-Driven Similarity-Clarity Tradeoff for Image Super-Resolution Quality Assessment
 ## Abstract
Super-Resolution (SR) algorithms aim to enhance the resolutions of images. Massive deep-learning-based SR techniques have emerged in recent years. In such case, a visually appealing output may contain additional details compared with its reference image. Accordingly, fully referenced Image Quality Assessment (IQA) cannot work well; however, reference information remains essential for evaluating the qualities of SR images. This poses a challenge to SR-IQA: How to balance the referenced and no-reference scores for user perception? In this paper, we propose a Perception-driven Similarity-Clarity Tradeoff (PSCT) model for SR-IQA. Specifically, we investigate this problem from both referenced and no-reference perspectives, and design two deep-learning-based modules to obtain referenced and no-reference scores. We present a theoretical analysis based on Human Visual System (HVS) properties on their tradeoff and also calculate adaptive weights for them. Experimental results indicate that our PSCT model is superior to the state-of-the-arts on SR-IQA. In addition, the proposed PSCT model is also capable of evaluating quality scores in other image enhancement scenarios, such as deraining, dehazing and underwater image enhancement.
## PSCT code
## Enviroment  
python 3.7
tensorflow 2.5.0
## 1. Preprocessing
For training, run ./preprocess/makemat_trainandtest.m  
For testing only, run ./preprocess/makemat_onlyfortest.m
## 2. Training and testing
Run main_trainandtest.py
## 3. Test using a pre-trained model
Run main_onlyfortest.py  
The pre-trained PSCT models can be downloaded at https://drive.google.com/drive/folders/1GprXj3dlXLaiFOzUUbEbuCkf8o1MjHwc  

If you have questions, please contact kekezhang1102@163.com (recommend) or 201110007@fzu.edu.cn.
