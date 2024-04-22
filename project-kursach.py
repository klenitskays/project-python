from flask import Flask, render_template
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot

app = Flask(__name__)
app.static_folder = 'static'

@app.route('/')
def index():
    # Загрузка данных
    data = pd.read_csv('online_shoppers_intention.csv')

    # Подсчет количества визитов или продуктовых запросов по месяцам
    monthly_data = data.groupby('Month').size()

    # Создание столбчатой диаграммы с использованием Plotly
    bar_data = go.Bar(x=monthly_data.index, y=monthly_data.values)

    bar_layout = go.Layout(
        title='Столбчатая диаграмма по месяцам',
        xaxis=dict(title='Месяц'),
        yaxis=dict(title='Количество')
    )

    fig = go.Figure(data=[bar_data], layout=bar_layout)

    # Преобразование диаграммы в HTML
    plot_div = plot(fig, output_type='div')

    # Отображение результатов на веб-странице и передача данных
    return render_template('index.html', plot_div=plot_div, data=data.to_csv(index=False))

if __name__ == '__main__':
    app.run(debug=True, port=8000, threaded=True)