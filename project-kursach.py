from flask import Flask, render_template
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mpld3

app = Flask(__name__)

@app.route('/')
def index():
    # ЗАГРУЗКА ДАННЫХ
    data = pd.read_csv('online_shoppers_intention.csv')

    # Создание круговой диаграммы
    revenue_counts = data['Revenue'].value_counts()
    pie_chart = revenue_counts.plot(kind='pie', colors=sns.color_palette(), autopct='%1.1f%%')
    plt.axis('equal')
    plt.title('Доля совершенных покупок')

    # Преобразование круговой диаграммы в интерактивный формат
    pie_chart_html = mpld3.fig_to_html(pie_chart.figure)

    # Создание графика столбцов
    month_counts = data['Month'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=month_counts.index, y=month_counts.values, ax=ax)
    ax.set_xlabel('Месяц')
    ax.set_ylabel('Количество')
    ax.set_title('Количество посещений в разные месяцы')

    # Преобразование графика столбцов в интерактивный формат
    column_chart_html = mpld3.fig_to_html(fig)

    # Преобразование данных в HTML-таблицу
    table_html = data.head().to_html()

    # Отображение результатов на веб-странице
    return render_template('index.html', table=table_html, column_chart_html=column_chart_html, pie_chart_html=pie_chart_html)

if __name__ == '__main__':
    app.run()