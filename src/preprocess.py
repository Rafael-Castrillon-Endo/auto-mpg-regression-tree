FEATURES = ['cylinders','displacement',
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

def get_avarage_features(data):
    avarage = [0] * len(FEATURES)
    size = [0] * len(FEATURES)
    for i in range(len(data)):
        row = data.iloc[i]
        for j in range(len(FEATURES)):
            if(row[FEATURES[j]] != '?'):
                avarage[j] += float(row[FEATURES[j]])
                size[j] += 1
    for i in range(len(avarage)):
        avarage[i] = avarage[i] / size[i]
    return avarage


def preprocess(data):
    index = emtpy_values(data)
    #for i in range(len(index)):
     #   row = data.iloc[index[i]['index_value']]
      #  print(row)
       # print("================================")
    #print(f"size -> {len(index)}")
    if(len(index)):
        avarage = get_avarage_features(data)
        for i in range(len(index)):
            #{'index_value': x, 'col' : x}
            index_value = index[i]['index_value']
            col = index[i]['col']
            data.at[index_value, FEATURES[col]] = (avarage[col])
        return data
    else:
        return data


    

