import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import os


# --- 2. PREPARAÇÃO DO DATASET (COM MÚLTIPLOS ARQUIVOS) ---

# AQUI ESTÁ A MUDANÇA PRINCIPAL:
# Em vez de um único arquivo, passamos um padrão (wildcard) para carregar
# todos os arquivos .txt que estão dentro da pasta 'transcricao'.
caminho_dos_arquivos = "transcricao/*.txt"

# A biblioteca 'datasets' vai ler todos os arquivos que correspondem ao padrão
# e tratá-los como um único grande dataset. É muito eficiente.
dataset = load_dataset("text", data_files=caminho_dos_arquivos)

print("\nDataset carregado com múltiplos arquivos:")
print(dataset)
print(f"Número de exemplos (linhas/documentos): {len(dataset['train'])}")


# --- O RESTO DO CÓDIGO PERMANECE IGUAL ---

# --- 3. CONFIGURAÇÃO DO MODELO E TOKENIZADOR ---
model_name = "PY007/TinyLlama-1.1B-Chat-v0.3" # Substitua por um Llama se tiver acesso/hardware
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# --- 4. PROCESSAMENTO DOS DADOS (TOKENIZAÇÃO) ---
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# --- 5. TREINAMENTO ---
output_dir = "./resultado_treinamento_continuado"

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=1,             # Com mais dados, você pode precisar de menos épocas!
    per_device_train_batch_size=2,
    save_steps=100,                 # Ajuste os passos de salvamento para o novo tamanho do dataset
    save_total_limit=2,
    logging_steps=20,               # Ajuste os passos de log
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

print("\n--- INICIANDO O TREINAMENTO COM MÚLTIPLOS ARQUIVOS ---")
trainer.train()
print("--- TREINAMENTO CONCLUÍDO ---")

# --- 6. SALVAR E TESTAR O MODELO ---
final_model_path = os.path.join(output_dir, "final_model")
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"Modelo final salvo em: {final_model_path}")

prompt_inicial = "Na minha opinião, a coisa mais importante sobre código é"
inputs = tokenizer(prompt_inicial, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50, num_return_sequences=1)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n--- TESTANDO O MODELO TREINADO ---")
print(f"Prompt: {prompt_inicial}")
print(f"Texto Gerado: {generated_text}")