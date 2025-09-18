import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

print("Загрузка датасета Википедии (новая версия)...")
dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")

print(f"Всего статей: {len(dataset)}")
print(f"Пример статьи:\n{dataset[0]['text'][:500]}...\n")

# ============================
# 2. Инициализация токенизатора
# ============================

MODEL_NAME = "gpt2"  # или "bert-base-uncased", "EleutherAI/gpt-neo-125M" и т.д.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Установим pad_token, если его нет (например, у GPT-2 его нет по умолчанию)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================
# 3. Предобработка текста и токенизация
# ============================

MAX_LENGTH = 512  # длина последовательности (можно изменить)

def tokenize_function(examples):
    # Токенизируем текст с добавлением EOS и паддингом до MAX_LENGTH
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_overflowing_tokens=True,  # разбивает длинные тексты на части
        stride=128,  # перекрытие между частями для лучшей контекстуализации
        return_tensors="pt"
    )

print("Токенизация данных...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text", "title", "id"],  # удаляем ненужные колонки
    desc="Токенизация"
)

# Оставим только input_ids и attention_mask
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

print(f"Токенизировано примеров: {len(tokenized_dataset)}")

# ============================
# 4. Создание PyTorch Dataset
# ============================

class WikiDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = item['input_ids']
        # Для языковой модели: target = input_ids (сдвиг внутри модели)
        return {
            'input_ids': input_ids,
            'labels': input_ids.clone()  # для LM loss (labels = input_ids)
        }

# ============================
# 5. DataLoader
# ============================

BATCH_SIZE = 8  # подберите под вашу GPU память
train_dataset = WikiDataset(tokenized_dataset)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

print(f"DataLoader готов. Размер батча: {BATCH_SIZE}")

# ============================
# 6. Пример использования в обучении
# ============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# Пример: получение одного батча
for batch in train_loader:
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    print(f"input_ids shape: {input_ids.shape}")
    print(f"labels shape: {labels.shape}")
    break  # только для демонстрации

print("✅ Данные готовы для обучения LLM!")