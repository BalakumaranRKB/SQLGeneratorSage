**SQLGeneratorSage: Insights from a  LLaMA Model(meta-llama/Llama-2-7b-chat-hf) trained on synthetic SQL generated data**

This is the repo for the Gen-AI project which aims to build and generate SQL queries based on the prompt provided to the Llama model. Finetuning(mainly using LoRA or QLoRA) was performed to check if there was an improvement in the performance of the model. 

  
  The repo contains:
    * The 80k data used for used fine-tuning the model.
	* The code for generating the data in a format acceptable by the model
	* The adapter/fine-tuned model folder having weights that can be reused with a Llama-2-7b-chat-hf model to help generate SQL queries.  
 
  Overview:

  In this project, I fine-tuned a Llama-2-7b-chat-hf model with 7 billion parameters on a T4 GPU with high RAM using Google Colab. I noticed that a T4 only has 16 GB of VRAM, which was barely enough to store Llama 2–7b's weights (7b × 2 bytes = 14 GB in FP16). Additionally, I had to consider the overhead due to optimizer states, gradients, and forward activations. This meant that a full fine-tuning was not possible: I needed to use parameter-efficient fine-tuning (PEFT) techniques like LoRA or QLoRA.
  
  I fine-tuned the model in 4-bit precision to drastically reduce the VRAM usage, which is why I used QLoRA. I leveraged the Hugging Face ecosystem with the transformers, accelerate, peft, trl, and bitsandbytes libraries. I implemented this using code based on Younes Belkada's GitHub Gist. First, I installed and loaded these libraries.



  About the Dataset used:

  gretelai/synthetic_text_to_sql is a large, diverse synthetic Text-to-SQL dataset created with Gretel Navigator, available under Apache 2.0 license.
  Key features:

* 105,851 records (100,000 train, 5,851 test)
* ~23M total tokens, ~12M SQL tokens
* 100 distinct domains
* Covers various SQL tasks and complexity levels
* Includes database context, explanations, and training tags

  As of April 2024, it's the largest synthetic text-to-SQL dataset available. This dataset showcases Gretel's capabilities in creating tailored synthetic data for specific use cases and advancing data-centric AI.


  Dataset Link: https://huggingface.co/datasets/gretelai/synthetic_text_to_sql
