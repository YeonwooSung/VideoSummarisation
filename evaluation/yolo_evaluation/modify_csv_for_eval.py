import pandas as pd


top10_classes = [
    'rice', 'beef-curry', 'sushi', 
    'fried-rice', 'toast', 'hamburger', 
    'sandwiches', 'ramen-noodle', 'miso-soup', 'egg-sunny-side-up'
]


f = open('detected_food.csv', 'r')
f_out = open('detected_food_top10.csv', 'w')

isInit = True

for l in f:
    line = l.strip()
    if isInit:
        f_out.write(line)
        f_out.write('\n')
        isInit = False
        continue
    splitted = line.split(',')
    newStr = splitted[0] + ','

    if splitted[1] in top10_classes:
        newStr += splitted[1]
        newStr += ','
    else:
        newStr += 'other,'

    if splitted[2] in top10_classes:
        print(splitted[2])
        newStr += splitted[2]
    else:
        newStr += 'other'

    newStr += '\n'
    f_out.write(newStr)

f_out.close()
f.close()


df = pd.read_csv('detected_food.csv')
row_for_other = df.loc[df['object'] == 'other']
#TODO df['object'] == 'other' 이 참인 행들을 모두 가져와서, 이 행들의 수를 1/5 (예시) 로 줄이기
#TODO 그 결과를 데이터프레임에 적용시키기!