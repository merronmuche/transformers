import tensorflow as tf

from src.transformers.models.phi3.modeling_tf import Phi3Config, TFPhi3ForCausalLM
from transformers import CONFIG_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING, AutoTokenizer, TFAutoModelForCausalLM


# Register the custom configuration and model
CONFIG_MAPPING.update({"phi3": Phi3Config})
MODEL_FOR_CAUSAL_LM_MAPPING.update({Phi3Config: TFPhi3ForCausalLM})

# Define the TensorFlow device
tf_device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

def run_phi3():
    input_text = ["""
         <start_of_turn>user
         Please paraphrase the word "jumps" in the following sentence: "The quick brown fox jumps over the lazy dog."
         <end_of_turn>
         <start_of_turn>model
         The quick brown fox leaps over the lazy dog.
         <end_of_turn>

         <start_of_turn>user
         Please paraphrase the word "effective" in the following sentence: "Effective communication is key to successful teamwork."
         <end_of_turn>
         <start_of_turn>model
         Efficient communication is key to successful teamwork.
         <end_of_turn>

         <start_of_turn>user
         Please paraphrase the word "revolutionized" in the following sentence: "Technology has revolutionized the way we live and work."
         <end_of_turn>
         <start_of_turn>model
         Technology has transformed the way we live and work.
         <end_of_turn>
         """]

    model_id = "microsoft/Phi-3-mini-4k-instruct"

    with tf.device(tf_device):
        model = TFAutoModelForCausalLM.from_pretrained(model_id, from_pt=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(input_text, return_tensors="tf", padding=True)

        output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        print("Output from Phi3:\n", output_text)

run_phi3()
 