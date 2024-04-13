from flask import Flask, render_template
import pandas as pd
from sklearn.linear_model import LogisticRegression
import plotly.graph_objs as go
from plotly.subplots import make_subplots

app = Flask(__name__)

@app.route('/')
def index():
    # Загрузка данных
    data = pd.read_csv('online_shoppers_intention.csv')

    # Выделение признаков и целевой переменной
    X = data[['Administrative', 'Informational']]
    y = data['Revenue']

    # Обучение модели логистической регрессии
    model = LogisticRegression()
    model.fit(X, y)

    # Вычисление предсказаний модели
    predictions = model.predict(X)

    # Создание розового графика
    pink_line = [5, 10, 8, 3, 6, 9, 4]
    plt.plot(pink_line, color='pink')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Розовый график')

    # Преобразование графика столбцов и розового графика в интерактивный формат
    column_chart_html = mpld3.fig_to_html(fig)

    # Создание графика
    fig = make_subplots(rows=2, cols=1, subplot_titles=['Revenue', 'Prediction'])

    # График для реальных значений
    fig.add_trace(go.Scatter(x=data.index, y=data['Revenue'], mode='markers', name='Revenue'), row=1, col=1)

    # График для предсказаний
    fig.add_trace(go.Scatter(x=data.index, y=data['Prediction'], mode='markers', name='Prediction'), row=2, col=1)

    # Настройка макета графика
    fig.update_layout(height=600, width=800, showlegend=False)

    # Получение HTML-кода для встроенного графика
    graph_html = fig.to_html(full_html=False)

    # Отображение результатов на веб-странице
    return render_template('index.html', graph=graph_html)

if __name__ == '__main__':
    app.run()