"""Physical-memory acceptance and one frozen, supported QLoRA loader."""

def disposition(*, backward_complete, allocated, reserved, physical, offload):
    if max(allocated, reserved) >= physical:
        return 'TRAINING_MEMORY_INFEASIBLE'
    if not backward_complete or offload:
        return 'TRAINING_MECHANICS_FAILURE'
    return 'TRAINABLE_12GB'


def load(local, profile):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(local, local_files_only=True,
        quantization_config=quant, dtype=torch.bfloat16, device_map={'':'cuda:0'},
        attn_implementation='sdpa')
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False})
    model = get_peft_model(model, LoraConfig(r=profile['rank'], lora_alpha=profile['alpha'],
        lora_dropout=profile['dropout'], target_modules=profile['target_modules'],
        bias='none', task_type='CAUSAL_LM'))
    model.config.use_cache=False
    return model
