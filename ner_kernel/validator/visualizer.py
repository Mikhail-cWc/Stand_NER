import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple


class NERVisualizer:

    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'whitegrid'):
        self.figsize = figsize
        self.style = style
        sns.set_style(style)

    def _prepare_metrics_dataframe(self,
                                   results: Dict[str, Dict],
                                   model_name: str = "Model") -> pd.DataFrame:
        micro_metrics = results.get('micro_avg', {})
        macro_metrics = results.get('macro_avg', {})

        metrics_data = {
            'Model': model_name,
            'Metric Type': ['Micro Average', 'Macro Average'],
            'Precision': [micro_metrics.get('precision', 0), macro_metrics.get('precision', 0)],
            'Recall': [micro_metrics.get('recall', 0), macro_metrics.get('recall', 0)],
            'F1': [micro_metrics.get('f1', 0), macro_metrics.get('f1', 0)]
        }

        df = pd.DataFrame(metrics_data)

        label_metrics = results.get('label_metrics', {})
        label_rows = []

        for label, metrics in label_metrics.items():
            label_rows.append({
                'Model': model_name,
                'Metric Type': f'Label: {label}',
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1': metrics.get('f1', 0),
                'Support': metrics.get('support', 0)
            })

        if label_rows:
            labels_df = pd.DataFrame(label_rows)
            df = pd.concat([df, labels_df], ignore_index=True)

        return df

    def _prepare_multi_model_dataframe(self,
                                       results_dict: Dict[str, Dict]) -> pd.DataFrame:
        all_dfs = []

        for model_name, results in results_dict.items():
            model_df = self._prepare_metrics_dataframe(results, model_name)
            all_dfs.append(model_df)

        return pd.concat(all_dfs, ignore_index=True)

    def plot_model_comparison(self,
                              results_dict: Dict[str, Dict],
                              metric: str = 'F1',
                              figsize: Optional[Tuple[int, int]] = None,
                              title: str = "Сравнение NER моделей") -> plt.Figure:
        df = self._prepare_multi_model_dataframe(results_dict)

        avg_df = df[df['Metric Type'].isin(['Micro Average', 'Macro Average'])]

        fig, ax = plt.subplots(figsize=figsize or self.figsize)

        bar_width = 0.35
        index = np.arange(len(results_dict))

        micro_mask = avg_df['Metric Type'] == 'Micro Average'
        macro_mask = avg_df['Metric Type'] == 'Macro Average'

        models = sorted(df['Model'].unique())
        micro_values = [avg_df[(avg_df['Model'] == model) & micro_mask][metric].values[0]
                        if not avg_df[(avg_df['Model'] == model) & micro_mask].empty else 0
                        for model in models]

        macro_values = [avg_df[(avg_df['Model'] == model) & macro_mask][metric].values[0]
                        if not avg_df[(avg_df['Model'] == model) & macro_mask].empty else 0
                        for model in models]

        ax.bar(index - bar_width/2, micro_values, bar_width, label=f'Micro {metric}')
        ax.bar(index + bar_width/2, macro_values, bar_width, label=f'Macro {metric}')

        ax.set_xlabel('Модели')
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.set_xticks(index)
        ax.set_xticklabels(models)
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_label_performance(self,
                               results: Dict,
                               model_name: str = "Model",
                               figsize: Optional[Tuple[int, int]] = None,
                               title: str = "Производительность по меткам") -> plt.Figure:
        df = self._prepare_metrics_dataframe(results, model_name)

        label_df = df[df['Metric Type'].str.startswith('Label:')]

        if label_df.empty:
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
            ax.text(0.5, 0.5, "Нет данных о метриках по меткам",
                    horizontalalignment='center', verticalalignment='center')
            plt.tight_layout()
            return fig

        label_df['Label'] = label_df['Metric Type'].str.replace('Label: ', '')

        long_df = pd.melt(
            label_df,
            id_vars=['Label', 'Support'],
            value_vars=['Precision', 'Recall', 'F1'],
            var_name='Metric',
            value_name='Value'
        )

        fig, ax = plt.subplots(figsize=figsize or self.figsize)

        sns.barplot(data=long_df, x='Label', y='Value', hue='Metric', ax=ax)

        for i, label in enumerate(label_df['Label']):
            support = label_df[label_df['Label'] == label]['Support'].values[0]
            ax.text(i, -0.05, f'n={support}', ha='center', rotation=0, fontsize=9)

        ax.set_title(f"{title} - {model_name}")
        ax.set_xlabel('Метка')
        ax.set_ylabel('Значение метрики')

        plt.tight_layout()
        return fig

    def plot_confusion_heatmap(self,
                               results: Dict,
                               figsize: Optional[Tuple[int, int]] = None,
                               title: str = "Матрица ошибок по меткам") -> plt.Figure:
        if 'confusion_matrix' not in results:
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
            ax.text(0.5, 0.5, "Матрица ошибок отсутствует в результатах",
                    horizontalalignment='center', verticalalignment='center')
            plt.tight_layout()
            return fig

        confusion_matrix = results['confusion_matrix']
        labels = list(results.get('label_metrics', {}).keys())

        fig, ax = plt.subplots(figsize=figsize or self.figsize)
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax)

        ax.set_title(title)
        ax.set_xlabel('Предсказанная метка')
        ax.set_ylabel('Истинная метка')

        plt.tight_layout()
        return fig

    def plot_document_performance(self,
                                  results: Dict,
                                  top_n: int = 10,
                                  worst: bool = False,
                                  figsize: Optional[Tuple[int, int]] = None,
                                  title: Optional[str] = None) -> plt.Figure:

        if 'documents' not in results or not results['documents']:
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
            ax.text(0.5, 0.5, "Нет данных о производительности на уровне документов",
                    horizontalalignment='center', verticalalignment='center')
            plt.tight_layout()
            return fig

        doc_df = pd.DataFrame(results['documents'])
        doc_df = doc_df.sort_values(by='f1', ascending=worst)
        doc_df = doc_df.head(top_n)
        long_df = pd.melt(
            doc_df,
            id_vars=['document_id', 'support'],
            value_vars=['precision', 'recall', 'f1'],
            var_name='Metric',
            value_name='Value'
        )
        fig, ax = plt.subplots(figsize=figsize or self.figsize)
        sns.barplot(data=long_df, x='document_id', y='Value', hue='Metric', ax=ax)

        for i, doc_id in enumerate(doc_df['document_id']):
            support = doc_df[doc_df['document_id'] == doc_id]['support'].values[0]
            ax.text(i, -0.05, f'n={support}', ha='center', rotation=0, fontsize=9)
        plt.xticks(rotation=45, ha='right')
        if title is None:
            title_type = "худших" if worst else "лучших"
            title = f"Производительность {top_n} {title_type} документов"

        ax.set_title(title)
        ax.set_xlabel('Документ')
        ax.set_ylabel('Значение метрики')

        plt.tight_layout()
        return fig

    def create_dashboard(self,
                         results_dict: Dict[str, Dict],
                         include_docs: bool = True,
                         save_path: Optional[str] = None) -> None:
        n_models = len(results_dict)

        n_rows = 2 + (1 if include_docs else 0)
        n_cols = max(n_models, 1)

        fig = plt.figure(figsize=(n_cols * 6, n_rows * 5))

        if n_models > 1:
            ax1 = plt.subplot2grid((n_rows, n_cols), (0, 0), colspan=n_cols)
            self._plot_model_comparison_subplot(results_dict, ax=ax1)

        for i, (model_name, results) in enumerate(results_dict.items()):
            ax2 = plt.subplot2grid((n_rows, n_cols), (1, i % n_cols))
            self._plot_label_performance_subplot(results, model_name, ax=ax2)

        if include_docs:
            for i, (model_name, results) in enumerate(results_dict.items()):
                if 'documents' in results and results['documents']:
                    ax3 = plt.subplot2grid((n_rows, n_cols), (2, i % n_cols))
                    self._plot_document_performance_subplot(results, ax=ax3,
                                                            title=f"Топ-5 документов - {model_name}")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def _plot_model_comparison_subplot(self,
                                       results_dict: Dict[str, Dict],
                                       ax: plt.Axes,
                                       metric: str = 'F1') -> None:
        df = self._prepare_multi_model_dataframe(results_dict)

        avg_df = df[df['Metric Type'].isin(['Micro Average', 'Macro Average'])]

        bar_width = 0.35
        index = np.arange(len(results_dict))

        micro_mask = avg_df['Metric Type'] == 'Micro Average'
        macro_mask = avg_df['Metric Type'] == 'Macro Average'

        models = sorted(df['Model'].unique())

        micro_values = [avg_df[(avg_df['Model'] == model) & micro_mask][metric].values[0]
                        if not avg_df[(avg_df['Model'] == model) & micro_mask].empty else 0
                        for model in models]

        macro_values = [avg_df[(avg_df['Model'] == model) & macro_mask][metric].values[0]
                        if not avg_df[(avg_df['Model'] == model) & macro_mask].empty else 0
                        for model in models]

        ax.bar(index - bar_width/2, micro_values, bar_width, label=f'Micro {metric}')
        ax.bar(index + bar_width/2, macro_values, bar_width, label=f'Macro {metric}')

        ax.set_xlabel('Модели')
        ax.set_ylabel(metric)
        ax.set_title(f"Сравнение моделей по {metric}")
        ax.set_xticks(index)
        ax.set_xticklabels(models)
        ax.legend()

    def _plot_label_performance_subplot(self,
                                        results: Dict,
                                        model_name: str,
                                        ax: plt.Axes) -> None:
        df = self._prepare_metrics_dataframe(results, model_name)

        label_df = df[df['Metric Type'].str.startswith('Label:')]

        if label_df.empty:
            ax.text(0.5, 0.5, "Нет данных о метриках по меткам",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
            return

        label_df['Label'] = label_df['Metric Type'].str.replace('Label: ', '')

        long_df = pd.melt(
            label_df,
            id_vars=['Label', 'Support'],
            value_vars=['Precision', 'Recall', 'F1'],
            var_name='Metric',
            value_name='Value'
        )

        sns.barplot(data=long_df, x='Label', y='Value', hue='Metric', ax=ax)

        for i, label in enumerate(label_df['Label']):
            support = label_df[label_df['Label'] == label]['Support'].values[0]
            ax.text(i, -0.05, f'n={support}', ha='center', rotation=0, fontsize=8)

        ax.set_title(f"Метрики по меткам - {model_name}")
        ax.set_xlabel('Метка')
        ax.set_ylabel('Значение метрики')
        ax.legend(loc='upper right')

        if len(label_df) > 5:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    def _plot_document_performance_subplot(self,
                                           results: Dict,
                                           ax: plt.Axes,
                                           top_n: int = 5,
                                           title: str = "Топ документов") -> None:
        if 'documents' not in results or not results['documents']:
            ax.text(0.5, 0.5, "Нет данных о производительности документов",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)

            return

        doc_df = pd.DataFrame(results['documents'])

        doc_df = doc_df.sort_values(by='f1', ascending=False).head(top_n)

        long_df = pd.melt(
            doc_df,
            id_vars=['document_id', 'support'],
            value_vars=['precision', 'recall', 'f1'],
            var_name='Metric',
            value_name='Value'
        )

        sns.barplot(data=long_df, x='document_id', y='Value', hue='Metric', ax=ax)

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        ax.set_title(title)
        ax.set_xlabel('Документ')
        ax.set_ylabel('Значение метрики')
        ax.legend(loc='upper right')


def example_usage():
    """
    Пример использования класса NERVisualizer.
    """
    # Пример результатов для двух моделей
    model1_results = {
        'micro_avg': {'precision': 0.85, 'recall': 0.82, 'f1': 0.83, 'support': 500},
        'macro_avg': {'precision': 0.83, 'recall': 0.80, 'f1': 0.81},
        'label_metrics': {
            'PERSON': {'precision': 0.90, 'recall': 0.88, 'f1': 0.89, 'support': 200},
            'ORGANIZATION': {'precision': 0.82, 'recall': 0.80, 'f1': 0.81, 'support': 150},
            'LOCATION': {'precision': 0.78, 'recall': 0.72, 'f1': 0.75, 'support': 100},
            'DATE': {'precision': 0.85, 'recall': 0.80, 'f1': 0.82, 'support': 50}
        },
        'documents': [
            {'document_id': 'doc1', 'precision': 0.95, 'recall': 0.90, 'f1': 0.92, 'support': 50},
            {'document_id': 'doc2', 'precision': 0.88, 'recall': 0.85, 'f1': 0.86, 'support': 40},
            {'document_id': 'doc3', 'precision': 0.80, 'recall': 0.75, 'f1': 0.77, 'support': 30},
            {'document_id': 'doc4', 'precision': 0.70, 'recall': 0.65, 'f1': 0.67, 'support': 20},
            {'document_id': 'doc5', 'precision': 0.60, 'recall': 0.55, 'f1': 0.57, 'support': 10}
        ]
    }

    model2_results = {
        'micro_avg': {'precision': 0.80, 'recall': 0.78, 'f1': 0.79, 'support': 500},
        'macro_avg': {'precision': 0.78, 'recall': 0.75, 'f1': 0.76},
        'label_metrics': {
            'PERSON': {'precision': 0.85, 'recall': 0.82, 'f1': 0.83, 'support': 200},
            'ORGANIZATION': {'precision': 0.78, 'recall': 0.75, 'f1': 0.76, 'support': 150},
            'LOCATION': {'precision': 0.72, 'recall': 0.68, 'f1': 0.70, 'support': 100},
            'DATE': {'precision': 0.80, 'recall': 0.75, 'f1': 0.77, 'support': 50}
        },
        'documents': [
            {'document_id': 'doc1', 'precision': 0.90, 'recall': 0.85, 'f1': 0.87, 'support': 50},
            {'document_id': 'doc2', 'precision': 0.83, 'recall': 0.80, 'f1': 0.81, 'support': 40},
            {'document_id': 'doc3', 'precision': 0.75, 'recall': 0.70, 'f1': 0.72, 'support': 30},
            {'document_id': 'doc4', 'precision': 0.65, 'recall': 0.60, 'f1': 0.62, 'support': 20},
            {'document_id': 'doc5', 'precision': 0.55, 'recall': 0.50, 'f1': 0.52, 'support': 10}
        ]
    }

    # Создаем визуализатор
    visualizer = NERVisualizer()

    # Пример 1: Сравнение моделей
    models_dict = {
        'BERT-base': model1_results,
        'SpaCy-large': model2_results
    }

#     # Создаем график сравнения моделей
#     fig1 = visualizer.plot_model_comparison(models_dict)
#     plt.show()

#     # Пример 2: Визуализация производительности по меткам для первой модели
#     fig2 = visualizer.plot_label_performance(model1_results, model_name='BERT-base')
#     plt.show()

# # Пример 3: Визуализация лучших документов
#     fig3 = visualizer.plot_document_performance(model1_results, top_n=5)
#     plt.show()

#     # Пример 4: Визуализация худших документов
#     fig4 = visualizer.plot_document_performance(model1_results, top_n=5, worst=True)
#     plt.show()

    fig5 = visualizer.plot_confusion_heatmap(model1_results)
    plt.show()
    # Пример 5: Создание комплексного дашборда
    visualizer.create_dashboard(models_dict, save_path='ner_dashboard.png')


if __name__ == "__main__":
    example_usage()
