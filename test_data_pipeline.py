# test_data_pipeline.py (at project root)
from src.data import get_dataloaders


def main() -> None:
    # num_workers=0 and the __main__ guard avoid Windows spawn-multiprocessing hangs.
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=16, num_workers=0)

    for images, labels in train_loader:
        print(f"OK: batch loaded: {images.shape}")
        break


if __name__ == "__main__":
    main()
