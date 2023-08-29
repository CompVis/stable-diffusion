import torch


def main():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 0
    print(f"Successfully ran on {num_gpus} GPUs")


if __name__ == "__main__":
    main()
