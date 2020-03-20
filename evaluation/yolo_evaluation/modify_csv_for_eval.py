import pandas as pd


top10_classes = [
    'rice', 'beef-curry', 'sushi', 
    'fried-rice', 'toast', 'hamburger', 
    'sandwiches', 'ramen-noodle', 'miso-soup', 'egg-sunny-side-up'
]


f = open('detected_food.csv', 'r')
f_out = open('detected_food_top10.csv', 'w')

isInit = True

# preprocess data
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
        newStr += splitted[2]
    else:
        newStr += 'other'

    newStr += '\n'
    f_out.write(newStr)

# close file streams
f_out.close()
f.close()


# read csv file to dataframe
df = pd.read_csv('detected_food_top10.csv')

row_for_other = df.loc[df['object'] == 'other']
newSize = row_for_other.shape[0] // 8
new_sub_df = row_for_other.head(newSize)

new_df = df.loc[df['object'] != 'other']
new_df = new_df.append(new_sub_df)

# convert dataframe to csv file
new_df.to_csv('./detected_food_top10.csv', index=False)
