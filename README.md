# Assessment-Generation-and-Realtime-Feedback-System-using-Large-Language-Models 
Assessment Generation and real-time feedback System using (llama-2 7b chat hf) and prompt engineering.
More details related to implementation are available at the end of this file.

# Project description

![image](https://github.com/KamleshKumar427/Assessment-Generation-and-Realtime-Feedback-System-using-Large-Language-Models/assets/95052507/30b50de4-7cb9-4503-a23d-f06adb8a3eec)

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
![image](https://github.com/KamleshKumar427/Assessment-Generation-and-Realtime-Feedback-System-using-Large-Language-Models/assets/95052507/90883b42-e8d3-47b0-a93f-c103406b28f4)

# For frontend I have used Streamlit Python to just build a minimal web apps

# Interface/Frontend

![1](https://github.com/KamleshKumar427/Assessment-Generation-and-Realtime-Feedback-System-using-Large-Language-Models/assets/95052507/5f96107a-7f6d-4930-ab2b-76fb5ae63d4f)

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

![2](https://github.com/KamleshKumar427/Assessment-Generation-and-Realtime-Feedback-System-using-Large-Language-Models/assets/95052507/8bc8ca5e-4ad0-432e-9645-ecfe903a2374)

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

![3](https://github.com/KamleshKumar427/Assessment-Generation-and-Realtime-Feedback-System-using-Large-Language-Models/assets/95052507/2d71993b-0861-4d2e-a5f9-ea6cd2bb2869)

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

![4](https://github.com/KamleshKumar427/Assessment-Generation-and-Realtime-Feedback-System-using-Large-Language-Models/assets/95052507/d8acb180-00e0-422d-bbbd-2a681f538e1d)

Initially, we tried to fine tune our LLM model i.e. Llama-2 7b pretrained model, We started by
curating around 400 examples of teacher and student having conversation, mostly related to
parts of speech. And we were able to train the model with this data. And the model was able to
do conversation related to parts of speec. Results attache at the end of report.

But results we limited to this particular topic but very little data. And it was not feasible to curate
so much data. And, also for our case we require data related particular format i.e. MCQS, Short
Q/As, and then long feedback related to each solution.(Which was more or less not possible).
Also due to less computing resources, and time we made use of another popular technique
which is Prompt Engineering using the framework LangChain which is designed to simplify the
creation of applications using large language models, along with already fine tunned model
name Llama-2-7b-chat-hf,

For initial fine tunning we used various techniques such as Parameter Efficient Fine-Tuning
(PEFT), e.g. LoRA (Focuses on updating only a subset of the model's parameters resulting in
faster fine-tuning), and Quantization-Based Fine-Tuning (QLoRA), which involves reducing the
precision of model parameters resulting in lower GPU consumption etc. to reduce the model
size and to perform efficient fine tuning, but due to the following reasons could not continue:


● Limited resources. Training the Llama 7b model requires 16 GB of VRAM and inference
requires 7GB.


● Poor results on the collected dataset. Fine-tuning would require a large dataset which
would be difficult to collect in the given time frame.

