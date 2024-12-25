import joblib
import pandas as pd
import numpy as np
from TG import *

white_list = [1732460818, 1101170442, 1592432748, 1628414073, 1251217154, 1394050290, 1363803438, 1117628569, 1752300297,
              1001208105, 1338355106, 1233657442, 1009185002]


def get_tg_posts(file_name):

    global_count = 0
    posts_to_file = []

    client = get_connection()

    for channel in white_list:
        print(channel)
        message_id = 0
        count = 0
        while count < 20000:
            print(global_count)
            posts = get_posts(client, channel, datetime.now().strftime("%m/%d/%Y"), 300, message_id)

            if posts.messages:
                for message in posts.messages:
                    count += 1
                    message_id = message.id
                    text = message.message

                    if message_id == 0:
                        break

                    if text is None or len(text) < 1 or 'Реклама' in text:
                        continue

                    global_count += 1
                    text = text.replace(' ', ' ')
                    text = text.replace(' ', ' ')
                    text = text.replace('‌', ' ')
                    text = text.replace('​', ' ')
                    text = text.replace('­', '')
                    text = text.replace('🟥 Сайт / Подписаться', ' ')
                    text = text.replace('Мир Космоса. Подписаться', ' ')
                    text = text.replace('Подписаться на Жизнь на море | Предложить новость', ' ')
                    text = text.replace('Быть в курсе | Прислать новость', ' ')
                    text = text.replace('Типичная Анапа| Подписаться', ' ')
                    text = text.replace('😉Убежище Самара', ' ')
                    text = text.replace('❗️ Подписывайся на Mash', ' ')
                    text = text.replace('❤️ Подписывайся на Kub Mash', ' ')
                    text = text.replace('Океан 🌊', ' ')
                    text = text.replace('Подписывайтесь 🟥 @yandex', ' ')
                    text = text.replace('Подписывайтесь 〰️ @yandex', ' ')
                    text = text.replace('Подписывайтесь ✨ @yandex', ' ')
                    text = text.replace('Подписывайтесь 🔴 @yandex', ' ')

                    posts_to_file.append(text)
            else:
                break

    close_connection(client)
    print(global_count)
    df = pd.DataFrame(posts_to_file, columns=['review'])
    df.to_csv(file_name, index=False)


def merge_excel(filename1, filename2, filename3):
    df1 = pd.read_csv(filename1, sep='\t')
    df2 = pd.read_csv(filename2, sep='\t')
    df1_negative = df1[df1['sentiment'] == 'negative'].head(30000)
    df1_neutral = df1[df1['sentiment'] == 'neutral'].head(29000)
    df2_positive = df2[df2['sentiment'] == 'positive'].head(30000)
    result_df = pd.concat([df1_negative, df1_neutral, df2_positive])
    result_df.to_csv(filename3, index=False, sep='\t')


if __name__ == '__main__':

    filename = ''
    net_name = ''

    loaded_automl = joblib.load(net_name)

    new_data = pd.read_csv(filename, sep='\t')

    predictions = loaded_automl.predict(new_data)

    df = pd.DataFrame({
        'текст': new_data.iloc[:, 0],
        'предсказание': [''] * len(predictions.data)
    })

    for index, prediction in enumerate(predictions.data):
        max_index = np.argmax(prediction)
        if max_index == 0:
            result = 'Негативный'
        elif max_index == 1:
            result = 'Нейтральный'
        else:
            result = 'Позитивный'

        df.loc[index, 'предсказание'] = result

    print("Результаты предсказаний:")
    for _, row in df.iterrows():
        print(f"\nТекст:\n{row['текст']}\n\nРезультат предсказания:\n{row['предсказание']}")
        print("---")
