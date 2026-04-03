"""End-to-end NIDS pipeline: preprocess -> train -> evaluate -> save."""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocess import run_preprocessing
from src.train import train_all_models, save_best_model, print_comparison_table
from src.evaluate import generate_all_plots


def main():
    parser = argparse.ArgumentParser(description="NIDS ML Pipeline")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to directory containing dataset CSV files")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["unsw_nb15", "cicids2018", "cicids2017"],
                        help="Dataset to use (default: unsw_nb15)")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip generating visualization plots")
    args = parser.parse_args()

    # Step 1: Preprocess
    print("\n" + "=" * 60)
    print(" NETWORK INTRUSION DETECTION SYSTEM ")
    print("=" * 60)

    X_train, X_test, y_train, y_test, feature_names, class_names, scaler, imputer = (
        run_preprocessing(args.data_dir, args.dataset)
    )

    # Step 2: Train all models
    print("\n" + "=" * 60)
    print("STEP 5: Training models")
    print("=" * 60)
    results = train_all_models(X_train, y_train, X_test, y_test)

    # Step 3: Compare
    print_comparison_table(results)

    # Step 4: Save best model
    print("\n" + "=" * 60)
    print("STEP 6: Saving best model")
    print("=" * 60)
    save_best_model(results, feature_names, class_names, scaler, imputer)

    # Step 5: Generate plots
    if not args.skip_plots:
        print("\n" + "=" * 60)
        print("STEP 7: Generating evaluation plots")
        print("=" * 60)
        generate_all_plots(results, class_names)

    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETE ")
    print("=" * 60)
    print(f"\nTo serve the model as an API:")
    print(f"  uvicorn api.app:app --reload --host 0.0.0.0 --port 8000")
    print(f"  Then visit: http://localhost:8000/docs")


if __name__ == "__main__":
    main()
