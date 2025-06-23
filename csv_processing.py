import csv
import json
import sys

csv.field_size_limit(sys.maxsize)

with open('perfumes_table.csv', 'r') as file:
    csv_reader = csv.reader(file)

    header = next(csv_reader)
    print(f"Header: {header}")

    data = {}
    '''
    Header: ['rating', 'notes', 'designer', 'reviews', 'description', 'url', 'title']
    row[0]: 'rating'
    row[1]: 'notes'
    row[2]: 'designer'
    row[3]: 'reviews'
    row[4]: 'description'
    row[5]: 'url'
    row[6]: 'title'
    '''
    for row in csv_reader:
        #print(row)
        review = ""
        #print(row[3])
        if len(row[3]) >= 2:
            review = row[3]
            
        if row[6] not in data:
            data[row[6]] = {"description": row[4], 
                            "notes": row[1], 
                            "designer":row[2], 
                            "reviews": [review]}
            #data[row['title']] = [row['description'], row['notes'], row['designer'], [row['reviews']]]
        else:
            data[row[6]]["reviews"].append(review) #data[row['title']][3].append(row['reviews'])
    
    with open('data.json', 'w') as f:
        json.dump(data, f, indent = 4)
    
