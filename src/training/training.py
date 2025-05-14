import os
import argparse
from trainer import train_model


def main():
    """Main function to start the training process."""
    parser = argparse.ArgumentParser(
        description="Train waste classification model"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0005, help="Weight decay"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=10, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for dataloaders",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="waste-material-classification",
        help="Project name for W&B logging",
    )
    parser.add_argument(
        "--use_cross_validation",
        action="store_true",
        help="Use cross-validation",
    )
    parser.add_argument(
        "--no_cross_validation",
        dest="use_cross_validation",
        action="store_false",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of cross-validation splits",
    )
    parser.set_defaults(use_cross_validation=True)

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    # Train model with specified parameters
    model = train_model(
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        num_workers=args.num_workers,
        project_name=args.project_name,
        use_cross_validation=args.use_cross_validation,
        n_splits=args.n_splits,
    )

    print(f"Training complete. Model saved to output directory.")

    return model


if __name__ == "__main__":
    main()
