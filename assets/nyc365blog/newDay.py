import json
from datetime import date
path = '/Users/evcu/GitHub/evcu.github.io//assets/nyc365blog/data.json'

data = {}
newel = {}
newel[u'date'] = str(date.today())
newel[u'mood'] = unicode(raw_input("Enter mood -1/0/1:\n"))
print(newel)
newel[u'high'] = unicode(raw_input("Highlights\n"))
newel[u'low'] = unicode(raw_input("Lowlights:\n"))
newel[u'other'] = unicode(raw_input("Other:\n"))
newel[u'text'] = unicode(raw_input("Random Thoughts:\n"))

print(newel)
with open(path,'r') as data_file:
    print('Successfuly read')
    data = json.load(data_file)
    data.append(newel)
with open(path,'w') as data_file:
    data_file.write(json.dumps(data, sort_keys=True, indent=4))
    print('Successfuly written')
