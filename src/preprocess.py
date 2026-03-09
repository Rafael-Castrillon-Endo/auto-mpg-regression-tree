FEATURES = ['mpg', 'cylinders', 'displacement',
            'horsepower', 'weight', 'acceleration',
            'model year', 'origin']

def emtpy_values(data):
    index = []
    for i in range(len(data)):
        row = data.iloc[i]
        for j in range(len(FEATURES)):
            if(row[FEATURES[j]] == '?'):
                index.append({'index_value': i, 'col' : j})
    #print(len(index))
    return index

def preprocess(data):
    index = emtpy_values(data)
    #for i in range(len(index)):
     #   row = data.iloc[index[i]['index_value']]
      #  print(row)
       # print("================================")
    #print(f"size -> {len(index)}")
    

