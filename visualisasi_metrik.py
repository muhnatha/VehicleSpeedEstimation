import matplotlib.pyplot as plt
import numpy as np

# Data untuk Nilai Error Model pada Video Amplaz01a
labels_amplaz01a = ['MAE (Km/h)', 'RMSE (Km/h)', 'MAPE (Km/h)']
yolov11m_amplaz01a = [4.974, 6.338, 19.133]
yolov11n_amplaz01a = [6.117, 7.485, 24.135]

# Data untuk Nilai Error Model pada Video FKH01
labels_fkh01 = ['MAE (Km/h)', 'RMSE (Km/h)', 'MAPE (Km/h)']
yolov11m_fkh01 = [10.677, 13.192, 47.451]
yolov11n_fkh01 = [10.78, 12.059, 48.494]

# Data untuk Nilai Error Model pada Video FKH02
labels_fkh02 = ['MAE (Km/h)', 'RMSE (Km/h)', 'MAPE (Km/h)']
yolov11m_fkh02 = [6.603, 7.856, 26.576]
yolov11n_fkh02 = [12.28, 13.049, 50.632]

def plot_model_comparison(labels, data_m, data_n, title, model_m_name='YOLOv11m', model_n_name='YOLOv11n'):
    x = np.arange(len(labels))  
    width = 0.35  

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, data_m, width, label=model_m_name, color='skyblue')
    rects2 = ax.bar(x + width/2, data_n, width, label=model_n_name, color='lightcoral')

    ax.set_ylabel('Nilai Error')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height, 3)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.show()

# Membuat visualisasi untuk Nilai Error Model pada Video Amplaz01a
plot_model_comparison(labels_amplaz01a, yolov11m_amplaz01a, yolov11n_amplaz01a, 'Perbandingan Model pada Video Amplaz01a')

# Membuat visualisasi untuk Nilai Error Model pada Video FKH01
plot_model_comparison(labels_fkh01, yolov11m_fkh01, yolov11n_fkh01, 'Perbandingan Model pada Video FKH01')

# Membuat visualisasi untuk Nilai Error Model pada Video FKH02
plot_model_comparison(labels_fkh02, yolov11m_fkh02, yolov11n_fkh02, 'Perbandingan Model pada Video FKH02')