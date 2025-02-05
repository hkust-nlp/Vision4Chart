from llava.train.train_full_function import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
