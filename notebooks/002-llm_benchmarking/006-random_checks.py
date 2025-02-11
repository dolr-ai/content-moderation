# %%
# randomly check samples from latest benchmark results
import json
import os
import random


def check_random_results(output_dir: str, model_name: str, num_samples: int = 5):
    # Get the latest results file
    result_files = [
        f
        for f in os.listdir(output_dir)
        if f.startswith(f"{model_name}_benchmark_results_")
    ]
    if not result_files:
        print(f"No result files found in {output_dir}")
        return

    latest_file = max(
        result_files, key=lambda x: os.path.getctime(os.path.join(output_dir, x))
    )
    file_path = os.path.join(output_dir, latest_file)

    print(f"\nChecking random samples from: {latest_file}")
    print("-" * 80)

    # Load results
    results = []
    with open(file_path, "r") as f:
        for line in f:
            results.append(json.loads(line))

    # Randomly sample results
    sample_size = min(num_samples, len(results))
    random_samples = random.sample(results, sample_size)

    # Display samples
    for i, result in enumerate(random_samples, 1):
        print(f"\nSample {i}:")
        print(f"Text: {result['text'][:200]}...")
        print(f"Actual category: {result['actual_category']}")
        print(
            f"Predicted category: {result['processed_response']['predicted_category']}"
        )
        print(f"Confidence: {result['processed_response']['predicted_score']:.2f}")
        print(f"Processing time: {result['processing_time']:.2f} seconds")
        print("-" * 80)


check_random_results(
    output_dir="benchmark_results",
    model_name="llama32",
    num_samples=5,
)
