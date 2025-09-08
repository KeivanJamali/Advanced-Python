import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


class Visulaze:
    def __init__(self, output_path, plot_path):
        self.output_path = output_path
        self.plot_path = plot_path
        self._read_data()

    def _read_data(self):
        self.summary = pd.read_csv(self.output_path/"summary_file.csv", index_col=0)

    def analyze_conversation(self, hist: bool = False, bar: bool = False, scatter: bool = False, pair: bool = False, heatmap: bool = False):
        """
        df: DataFrame with columns
        ['id', 'name', 'refrence_id', 'total_messages', 'null_mean', 'delay_mean', 'duration_mean']
        
        This function creates multiple visualizations to analyze each person's messaging behavior.
        """
        sns.set(style="whitegrid")
        
        numeric_cols = ['total_messages', 'null_mean', 'delay_mean_hours', 'duration_mean_hours']

        # Suppose your columns are delay_mean and duration_mean
        self.summary['delay_mean_hours'] = pd.to_timedelta(self.summary['delay_mean']).dt.total_seconds() / 3600
        self.summary['duration_mean_hours'] = pd.to_timedelta(self.summary['duration_mean']).dt.total_seconds() / 3600

        if hist:
            # 1. Histograms for all numeric columns
            plt.figure(figsize=(16, 10))
            for i, col in enumerate(numeric_cols, 1):
                plt.subplot(2, 2, i)
                sns.histplot(self.summary[col], bins=15, kde=True, color='skyblue')
                plt.title(f'Distribution of {col}')
            plt.tight_layout()
            plt.savefig(self.plot_path/"histograms.png", bbox_inches='tight')
            plt.show()
        
        if bar:
            # 2. Bar plot: total messages per person
            plt.figure(figsize=(12,6))
            sns.barplot(x='id', y='total_messages', data=self.summary, hue='refrence_id', palette='viridis', dodge=False, legend=False)
            plt.title('Total Requests per Person')
            plt.xticks(rotation=45)
            plt.savefig(self.plot_path/"total_requests.png", bbox_inches='tight')
            # plt.show()
            
            # 3. Bar plot: % of null requests per person
            plt.figure(figsize=(12,6))
            sns.barplot(x='id', y='null_mean', data=self.summary, hue='refrence_id', palette='magma', dodge=False, legend=False)
            plt.title('% of Requests with No Reply per Person')
            plt.xticks(rotation=45)
            plt.savefig(self.plot_path/"null_requests.png", bbox_inches='tight')
            # plt.show()

        if scatter:
            # 4. Scatter: delay_mean vs duration_mean
            plt.figure(figsize=(8,6))
            sns.scatterplot(x='delay_mean_hours', y='duration_mean_hours', hue='id', data=self.summary, s=100, palette='tab10')
            plt.title('Mean Response Delay vs Total Duration')
            plt.xlabel('Average Delay (hours)')
            plt.ylabel('Average Duration (hours)')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.savefig(self.plot_path/"delay_vs_duration.png", bbox_inches='tight')
            # plt.show()

        if pair:   
            # 5. Pairplot for numeric columns
            sns.pairplot(self.summary[numeric_cols + ['id']], hue='id', height=2.5)
            plt.savefig(self.plot_path/"pairplot.png", bbox_inches='tight')
            # plt.show()
        
        if heatmap:
            # 6. Optional: heatmap of correlations
            plt.figure(figsize=(8,6))
            sns.heatmap(self.summary[numeric_cols].corr(), annot=True, cmap='coolwarm')
            plt.title('Correlation Heatmap of Numeric Features')
            plt.savefig(self.plot_path/"correlation_heatmap.png", bbox_inches='tight')
            # plt.show()

    def analyze_students(self):
        # Read all CSVs into one dataframe
        all_files = [
            os.path.join(self.output_path, f)
            for f in os.listdir(self.output_path)
            if f.endswith(".csv") and f.startswith("student")
        ]
        df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
        self.df = df
        print(len(df), len(df.drop_duplicates()))

        df['delay_hours'] = pd.to_timedelta(df['latency']).dt.total_seconds() / 3600
        df['duration_hours'] = pd.to_timedelta(df['duration']).dt.total_seconds() / 3600

        # -------------------------
        # Example analyses
        # -------------------------

        # 1. How many questions each student asked
        questions_per_student = df.groupby("national_id").size().sort_values(ascending=False)
        questions_per_student.to_csv(self.plot_path/"questions_per_student.csv", encoding="utf-8-sig")

        # 2. Most common subjects overall
        subjects_count = df["subject"].value_counts()
        subjects_count.to_csv(self.plot_path/"subjects_count.csv", encoding="utf-8-sig")

        # 3. Operator load (by reference_code)
        operator_load = df.groupby("refrence_code").size().sort_values(ascending=False)
        operator_load.to_csv(self.plot_path/"operator_load.csv", encoding="utf-8-sig")

        # 4. Average latency/duration per student
        latency_duration = df.groupby("national_id")[["delay_hours", "duration_hours"]].mean()
        latency_duration.to_csv(self.plot_path/"latency_duration.csv", encoding="utf-8-sig")

        # 5. Subjects with students spread into columns
        subjects_students_expanded = (
            df.groupby("subject")["national_id"]
            .unique()                # unique students per subject
            .apply(lambda ids: [str(int(x)) for x in ids])  # ensure str(int(...))
        )

        # Convert into a DataFrame with each ID in its own column
        max_len = subjects_students_expanded.apply(len).max()
        expanded_rows = {
            subj: ids + [""] * (max_len - len(ids))   # pad with empty strings
            for subj, ids in subjects_students_expanded.items()
        }
        expanded_df = pd.DataFrame.from_dict(expanded_rows, orient="index")

        # Put 'subject' as a column instead of index
        expanded_df.insert(0, "subject", expanded_df.index)
        expanded_df.reset_index(drop=True, inplace=True)

        expanded_df.to_csv(self.plot_path/"subjects_students_expanded.csv", encoding="utf-8-sig", index=False)

