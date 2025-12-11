from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

if __name__ == "__main__":
    print(f"Downloading model: {MODEL_NAME}")
    AutoTokenizer.from_pretrained(MODEL_NAME)
    AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    print("Model downloaded successfully.")
