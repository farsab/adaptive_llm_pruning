
import csv, os
import matplotlib.pyplot as plt

class MetricsLogger:
    def __init__(self, save_dir='results'):
        self.records = []
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def log(self, step, ece, nll, entropy):
        self.records.append({'step': step, 'ECE': ece, 'NLL': nll, 'Entropy': entropy})

    def save_plots(self):
        if not self.records:
            return
        import pandas as pd
        df = pd.DataFrame(self.records)
        df.to_csv(os.path.join(self.save_dir, 'metrics.csv'), index=False)
        plt.figure()
        plt.plot(df['step'], df['ECE'], label='ECE')
        plt.plot(df['step'], df['NLL'], label='NLL')
        plt.legend()
        plt.title('Calibration Metrics')
        plt.savefig(os.path.join(self.save_dir, 'calibration_plot.png'))

        plt.figure()
        plt.plot(df['step'], df['Entropy'], label='Attention Entropy')
        plt.legend()
        plt.title('Interpretability Proxy')
        plt.savefig(os.path.join(self.save_dir, 'attention_entropy_plot.png'))
