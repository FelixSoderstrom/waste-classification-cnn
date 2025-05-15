import os
import argparse
import time
from src.training.trainer import train_model
from src.evaluation.evaluation import evaluate_model
from src.evaluation.visualization import create_visualizations
from src.evaluation.utils import get_default_model_path
from src.training.utils import create_training_summary


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
        "--max_epochs", type=int, default=20, help="Maximum number of epochs"
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
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training and only run evaluation and visualization",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model for evaluation (only used with --skip_training)",
    )
    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="Skip evaluation and visualization (training only)",
    )
    parser.set_defaults(
        use_cross_validation=True, skip_training=False, skip_evaluation=False
    )

    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)

    trained_model_path = args.model_path

    training_start_time = time.time()

    if not args.skip_training:
        print("\n" + "=" * 40)
        print("STEP 1: TRAINING MODEL")
        print("=" * 40)

        trained_model = train_model(
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            num_workers=args.num_workers,
            project_name=args.project_name,
            use_cross_validation=args.use_cross_validation,
            n_splits=args.n_splits,
        )

        trained_model_path = get_default_model_path()
        print(f"\nTraining complete. Model saved to: {trained_model_path}")

        model_dir = os.path.dirname(trained_model_path)
        session_dir = model_dir

        if os.path.basename(model_dir).startswith("fold_"):
            session_dir = os.path.dirname(model_dir)

        create_training_summary(
            session_dir=session_dir,
            trained_model_path=trained_model_path,
            training_start_time=training_start_time,
            training_args=args,
            model=trained_model,
        )

    if not args.skip_evaluation and trained_model_path is not None:
        print("\n" + "=" * 40)
        print("STEP 2: EVALUATING MODEL")
        print("=" * 40)

        evaluation_results = evaluate_model(
            model_path=trained_model_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        if not args.skip_training:
            create_training_summary(
                session_dir=evaluation_results["session_dir"],
                trained_model_path=trained_model_path,
                training_start_time=training_start_time,
                training_args=args,
                evaluation_results=evaluation_results,
                model=evaluation_results["model"],
            )

        print("\n" + "=" * 40)
        print("STEP 3: CREATING VISUALIZATIONS")
        print("=" * 40)

        visualization_results = create_visualizations(
            evaluation_results=evaluation_results,
            model=evaluation_results["model"],
            device=evaluation_results["device"],
            session_dir=evaluation_results["session_dir"],
        )

        print("\nVisualizations saved:")
        for key, path in visualization_results.items():
            print(f"- {key}: {path}")

    print("\nProcess completed successfully!")

    return 0


if __name__ == "__main__":
    main()
