
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

class EDAUtils:
    """
    Utility class for performing Exploratory Data Analysis (EDA).
    Contains static methods for generating various plots.
    """

    @staticmethod
    def plot_distribution(df: pd.DataFrame, column: str, bins: int = 30) -> plt.Figure:
        """
        Plots the distribution of a numerical column.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            column (str): The name of the column to plot.
            bins (int): The number of bins for the histogram.

        Returns:
            plt.Figure: The matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df[column], bins=bins, edgecolor='black')
        ax.set_title(f'Distribution of {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        return fig

    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame, numerical_columns: List[str]) -> plt.Figure:
        """
        Plots a correlation heatmap for the given numerical columns.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            numerical_columns (List[str]): A list of numerical column names.

        Returns:
            plt.Figure: The matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        correlation_matrix = df[numerical_columns].corr()
        cax = ax.matshow(correlation_matrix, cmap='coolwarm')
        fig.colorbar(cax)
        
        ax.set_xticks(range(len(correlation_matrix.columns)))
        ax.set_yticks(range(len(correlation_matrix.columns)))
        ax.set_xticklabels(correlation_matrix.columns, rotation=90)
        ax.set_yticklabels(correlation_matrix.columns)
        
        ax.set_title('Correlation Heatmap', pad=20)
        return fig

    @staticmethod
    def plot_drive_config_distribution(df: pd.DataFrame) -> plt.Figure:
        """
        Plots the distribution of Drive Configurations.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.

        Returns:
            plt.Figure: The matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        drive_config_counts = df['drive_config'].value_counts()
        ax.bar(drive_config_counts.index, drive_config_counts.values)
        ax.set_title('Distribution of Drive Configurations')
        ax.set_xlabel('Drive Configuration')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        return fig
        
    @staticmethod
    def plot_top_makes(df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
        """
        Plots the top N EV makes.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            top_n (int): The number of top makes to display.

        Returns:
            plt.Figure: The matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        make_counts = df['make'].value_counts().nlargest(top_n)
        ax.bar(make_counts.index, make_counts.values, color='skyblue')
        ax.set_title(f'Top {top_n} Electric Vehicle Makes')
        ax.set_xlabel('Make')
        ax.set_ylabel('Number of Vehicles')
        ax.tick_params(axis='x', rotation=45)
        return fig
