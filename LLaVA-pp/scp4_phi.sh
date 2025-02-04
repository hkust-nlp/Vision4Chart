# cd LLaVA-pp
# Copy necessary files
cp Phi-3-V/train.py LLaVA/llava/train/train.py
cp Phi-3-V/llava_phi3.py LLaVA/llava/model/language_model/llava_phi3.py
cp Phi-3-V/builder.py LLaVA/llava/model/builder.py
cp Phi-3-V/model__init__.py LLaVA/llava/model/__init__.py
cp Phi-3-V/main__init__.py LLaVA/llava/__init__.py
cp Phi-3-V/conversation.py LLaVA/llava/conversation.py

# # Training commands
# cp scripts/Phi3-V_pretrain.sh LLaVA/Vi-phi3_pretrain.sh
# cp scripts/Phi3-V_finetune_lora.sh LLaVA/Vi-phi3_finetune_lora.sh