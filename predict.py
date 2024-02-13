from cog import BasePredictor, Input
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # load base LLM model, LoRA params and tokenizer
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            "neuralwork/mistral-7b-style-instruct",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            bnb_4bit_compute_dtype=torch.float16,
            load_in_4bit=True,
            cache_dir="hf-cache"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "neuralwork/mistral-7b-style-instruct", 
            cache_dir="hf-cache"
        )

    def format_instruction(self, input, event):
        return f"""You are a personal stylist recommending fashion advice and clothing combinations. Use the self body and style description below, combined with the event described in the context to generate 5 self-contained and complete outfit combinations.
            ### Input:
            {input}

            ### Context:
            I'm going to a {event}.

            ### Response:
        """
        
    def predict(
        self,
        prompt: str = Input(description="Self description of your body type and personal style"),
        event: str = Input(description="Event description"),
    ) -> str:
        """Run a single prediction on the model"""
        prompt = self.format_instruction(prompt, event)

        input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
        
        # inference
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=input_ids, 
                max_new_tokens=800, 
                do_sample=True, 
                top_p=0.9,
                temperature=0.9
            )
            
        # decode output tokens and strip response
        outputs = outputs.detach().cpu().numpy()
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output = outputs[0][len(prompt):]
        
        return output