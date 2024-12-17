import streamlit as st

def predict(question):
    import torch
    import transformers
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        HfArgumentParser,
        BitsAndBytesConfig,
        TrainingArguments,
        pipeline,
        logging,
    )
    import pandas as pd
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

    model_name = "NousResearch/Llama-2-7b-chat-hf"
    adapters_name = "/workspace/my_project/fine_tuned_model"

    print(f"Starting to load the model {model_name} into memory")

    try:
        # Load the base model
        m = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print(f"Combining  the model {model_name}  and adapter file into memory")

        # Load the adapter
        m = PeftModel.from_pretrained(m, adapters_name)

        # Merge the adapter with the base model
        m = m.merge_and_unload()

        # Load the tokenizer
        tok = LlamaTokenizer.from_pretrained(model_name)
        tok.bos_token_id = 1

        print(f"Successfully loaded the model {model_name} into memory")
    except Exception as e:
        print(f"An error occurred while loading the model: {str(e)}")


    def test_model(tokenizer, pipeline, prompt_to_test):
        """
        Perform a query
        print the result
        Args:
            tokenizer: the tokenizer
            pipeline: the pipeline
            prompt_to_test: the prompt
        Returns
            None
        """
        # adapted from https://huggingface.co/blog/llama2#using-transformers
        answer = []

        sequences = pipeline(
            prompt_to_test,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=1000,)
        for seq in sequences:
            response = seq['generated_text'].split('[/INST]')[-1].strip()
            answer.append(response)
        return answer

    import torch
    torch.cuda.empty_cache()


    def format_prompt(user_prompt):
      default_system_prompt="Answer the question to the best of your ability succintly. try atleast answering with whatever SQL knowledge(MS SQL server flavour) that you have and provide the SQL query. Do not give multiple answers."
      return f'''<s> [INST] <<SYS>> {default_system_prompt} <</SYS>> {user_prompt} [/INST]'''

    query_pipeline = transformers.pipeline("text-generation",model=m,tokenizer=tok,torch_dtype=torch.float16,device_map="auto",)

    return test_model(tok, query_pipeline, format_prompt(user_prompt))





# User input section
user_prompt = st.text_area(
        "Enter your prompt:",
        height=100,
        placeholder="Type your prompt here...",
        key="prompt_input"
    )

    
    # Generate button
if st.button("Generate Responses", key="generate"):
    if user_prompt:
        fine_tuned_model_LLM_answers = predict(user_prompt)
        print(fine_tuned_model_LLM_answers[0])
        st.session_state.fine_tuned_answer = fine_tuned_model_LLM_answers
    # Display output only if answers exist
    if fine_tuned_model_LLM_answers:
        st.markdown("### Fine-tuned LLM Output")
        st.write(fine_tuned_model_LLM_answers[0])
        
