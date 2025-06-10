import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from steadytext.models.loader import get_generator_model_instance
from steadytext.utils import LLAMA_CPP_GENERATION_SAMPLING_PARAMS_DETERMINISTIC, logger, DEFAULT_SEED
# set_deterministic_environment is called when utils is imported.

logger.setLevel("INFO")

print("Attempting to load generator model...")
model = get_generator_model_instance(force_reload=True)

if model:
    print("Generator model loaded. Attempting to generate text...")
    prompt = "Once upon a time"

    formatted_prompt = (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    sampling_params = LLAMA_CPP_GENERATION_SAMPLING_PARAMS_DETERMINISTIC.copy()

    try:
        output_dict = model(
            formatted_prompt,
            **sampling_params
        )
        print(f"Raw output dictionary: {output_dict}")

        generated_text = ""
        if output_dict and 'choices' in output_dict and            isinstance(output_dict['choices'], list) and len(output_dict['choices']) > 0 and            'text' in output_dict['choices'][0]:
            generated_text = output_dict['choices'][0]['text'].strip()
            print(f"Generated text: '{generated_text}'")
            if generated_text:
                 print("Text generation successful and output is not empty.")
            else:
                 print("Text generation produced empty output after stripping.")
                 sys.exit(1)
        else:
            print("Text generation failed or produced unexpected output structure.")
            sys.exit(1)

    except Exception as e:
        print(f"Error during generation: {e}")
        logger.error("Exception during generation", exc_info=True)
        sys.exit(1)
else:
    print("Failed to load generator model.")
    sys.exit(1)

print("Minimal generation test completed.")
