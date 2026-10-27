import pandas as pd
import ast
import re
import matplotlib.pyplot as plt
import argparse

# Function to extract numbers from cumulative confidence values
def extract_confidences(confidence_list):
    cleaned_list = []
    for confidence in confidence_list:
        if isinstance(confidence, (int, float)):
            cleaned_list.append(float(confidence))
        elif isinstance(confidence, str):
            # Handle "tensor(...)" strings
            match = re.search(r"tensor\(([\d\.]+)\)", confidence)
            if match:
                cleaned_list.append(float(match.group(1)))
            else:
                try:
                    cleaned_list.append(float(confidence))
                except ValueError:
                    pass
    return cleaned_list

def main():
    parser = argparse.ArgumentParser(description="Process and analyze cumulative confidence data.")
    parser.add_argument("--INPUT_PATH", default="matched_output.csv", help="Path to the input CSV file")
    parser.add_argument("--OUTPUT_PATH", default="processed_output.csv", help="Path to save the processed CSV file")
    args = parser.parse_args()

    # Load CSV data
    df = pd.read_csv(args.INPUT_PATH)

    # Convert 'Cumulative Confidences' from string representation of lists to actual lists
    df['Cumulative Confidences'] = df['Cumulative Confidences'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # Clean the 'Cumulative Confidences' to extract only numerical values
    df['Cumulative Confidences'] = df['Cumulative Confidences'].apply(
        lambda confs: extract_confidences(confs)
    )

    # # Mark hallucination based on the Score (1.0 = no hallucination, <1.0 = hallucination)
    # df['No Hallucination'] = df['Score'] == 1.0

    # # Define bins for confidence ranges
    # bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # bin_labels = [f"{bins[i]}~{bins[i+1]}" for i in range(len(bins)-1)]

    # # Get all unique stages in the dataset
    # unique_stages = df['Stage'].unique()

    # # Analyze and plot for each stage
    # for stage in unique_stages:
    #     print(f"Processing results for {stage}...")

    #     # Filter data for the current stage
    #     df_stage = df[df['Stage'] == stage]

    #     # Extract confidences for hallucinated and non-hallucinated cases
    #     hallucinated_confidences = [
    #         conf for confs in df_stage[~df_stage['No Hallucination']]['Cumulative Confidences'] for conf in confs
    #     ]
    #     no_halluc_confidences = [
    #         conf for confs in df_stage[df_stage['No Hallucination']]['Cumulative Confidences'] for conf in confs
    #     ]

    #     # Check if data is available for the current stage
    #     if not hallucinated_confidences and not no_halluc_confidences:
    #         print(f"No valid confidence data for {stage}. Skipping...")
    #         continue

    #     # Plot histogram for hallucinated cases
    #     if hallucinated_confidences:
    #         hist_halluc = pd.cut(
    #             hallucinated_confidences,
    #             bins=bins,
    #             labels=bin_labels,
    #             right=False,
    #             include_lowest=True
    #         ).value_counts()

    #         plt.figure(figsize=(8, 6))
    #         hist_halluc.sort_index().plot(kind="bar")
    #         plt.xlabel("Confidence Range")
    #         plt.ylabel("Number of Cases")
    #         plt.title(f"Confidence Levels with Hallucination ({stage})")
    #         plt.tight_layout()
    #         output_file_halluc = f"confidence_distribution_hallucinated_{stage.replace(' ', '_').lower()}.png"
    #         plt.savefig(output_file_halluc)
    #         print(f"Saved hallucination plot to {output_file_halluc}")
    #         plt.show()

    #     # Plot histogram for non-hallucinated cases
    #     if no_halluc_confidences:
    #         hist_no_halluc = pd.cut(
    #             no_halluc_confidences,
    #             bins=bins,
    #             labels=bin_labels,
    #             right=False,
    #             include_lowest=True
    #         ).value_counts()

    #         plt.figure(figsize=(8, 6))
    #         hist_no_halluc.sort_index().plot(kind="bar")
    #         plt.xlabel("Confidence Range")
    #         plt.ylabel("Number of Cases")
    #         plt.title(f"Confidence Levels Without Hallucination ({stage})")
    #         plt.tight_layout()
    #         output_file_no_halluc = f"confidence_distribution_nohalluc_{stage.replace(' ', '_').lower()}.png"
    #         plt.savefig(output_file_no_halluc)
    #         print(f"Saved non-hallucination plot to {output_file_no_halluc}")
    #         plt.show()
    # Mark hallucination based on the Score (1.0 = no hallucination, <1.0 = hallucination)
    df['No Hallucination'] = df['Score'] == 1.0

    # Define bins for confidence ranges
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bin_labels = [f"{bins[i]}~{bins[i+1]}" for i in range(len(bins)-1)]

    # Get all unique stages in the dataset
    unique_stages = df['Stage'].unique()

    # Initialize dataframes to store aggregated histogram data
    hist_halluc_aggregated = pd.DataFrame(index=bin_labels)
    hist_no_halluc_aggregated = pd.DataFrame(index=bin_labels)

    # Process each stage and aggregate data
    for stage in unique_stages:
        print(f"Processing results for {stage}...")

        # Filter data for the current stage
        df_stage = df[df['Stage'] == stage]

        # Extract confidences for hallucinated and non-hallucinated cases
        hallucinated_confidences = [
            conf for confs in df_stage[~df_stage['No Hallucination']]['Cumulative Confidences'] for conf in confs
        ]
        no_halluc_confidences = [
            conf for confs in df_stage[df_stage['No Hallucination']]['Cumulative Confidences'] for conf in confs
        ]

        # Create histograms for hallucinated cases
        if hallucinated_confidences:
            hist_halluc = pd.cut(
                hallucinated_confidences,
                bins=bins,
                labels=bin_labels,
                right=False,
                include_lowest=True
            ).value_counts()
            hist_halluc_aggregated[stage] = hist_halluc.sort_index()

        # Create histograms for non-hallucinated cases
        if no_halluc_confidences:
            hist_no_halluc = pd.cut(
                no_halluc_confidences,
                bins=bins,
                labels=bin_labels,
                right=False,
                include_lowest=True
            ).value_counts()
            hist_no_halluc_aggregated[stage] = hist_no_halluc.sort_index()

    # Fill missing values with 0
    hist_halluc_aggregated = hist_halluc_aggregated.fillna(0)
    hist_no_halluc_aggregated = hist_no_halluc_aggregated.fillna(0)

    # Plot aggregated histogram for hallucinated cases
    plt.figure(figsize=(10, 8))
    hist_halluc_aggregated.plot(kind="bar", width=0.8, figsize=(12, 8), stacked=False)
    plt.xlabel("Confidence Range")
    plt.ylabel("Number of Cases")
    plt.title("Confidence Levels with Hallucination (All Stages)")
    plt.legend(title="Stage", loc="upper right")
    plt.tight_layout()
    output_file_halluc_all = "confidence_distribution_hallucinated_all_stages.png"
    plt.savefig(output_file_halluc_all)
    print(f"Saved aggregated hallucination plot to {output_file_halluc_all}")
    plt.show()

    # Plot aggregated histogram for non-hallucinated cases
    plt.figure(figsize=(10, 8))
    hist_no_halluc_aggregated.plot(kind="bar", width=0.8, figsize=(12, 8), stacked=False)
    plt.xlabel("Confidence Range")
    plt.ylabel("Number of Cases")
    plt.title("Confidence Levels Without Hallucination (All Stages)")
    plt.legend(title="Stage", loc="upper right")
    plt.tight_layout()
    output_file_no_halluc_all = "confidence_distribution_nohalluc_all_stages.png"
    plt.savefig(output_file_no_halluc_all)
    print(f"Saved aggregated non-hallucination plot to {output_file_no_halluc_all}")
    plt.show()

    # Handle varying numbers of confidences dynamically
    max_confidences = df['Cumulative Confidences'].apply(len).max()
    confidence_column_names = [f'Cumulative Confidence {i+1}' for i in range(max_confidences)]

    # Expand the lists into columns
    confidences_df = pd.DataFrame(df['Cumulative Confidences'].tolist(), index=df.index, columns=confidence_column_names)
    df = pd.concat([df, confidences_df], axis=1)

    # Normalize 'Text Output' by removing non-alphanumeric characters for consistent comparison
    df['NormalizedTextOutput'] = df['Text Output'].apply(lambda x: re.sub(r'\W+', '', str(x)).lower())

    # Remove the 'Answer' column since it duplicates 'Ground Truth'
    df = df.drop(columns=['Answer'])

    # Ensure unique rows for each Doc ID and Stage combination (already inherently separate)
    df = df.drop_duplicates(subset=['Doc ID', 'Stage'], keep='first')

    # Compare text outputs across stages to identify changes
    df['Is Changed'] = df.groupby('Doc ID')['NormalizedTextOutput'].transform('nunique') > 1

    # Reorganize columns: Move confidence columns to the rightmost
    output_columns = [
        'Doc ID', 'Question', 'Ground Truth', 'Prediction', 'Score', 'Category', 'L2 Category', 'Stage',
        'Is Changed', 'Text Output', 'NormalizedTextOutput'
    ] + confidence_column_names

    # Ensure columns are in the correct order
    df = df.reindex(columns=output_columns)

    # Save the result to a new CSV file
    df.to_csv(args.OUTPUT_PATH, index=False)

    print(f"Processed data saved to '{args.OUTPUT_PATH}'")

if __name__ == "__main__":
    main()
