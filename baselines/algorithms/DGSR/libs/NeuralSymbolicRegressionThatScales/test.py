with open('requirements.txt', 'r') as fh:
    text = fh.readlines()

for l in text:
    l = l.replace('\n', '')
    o = l.split(' ')
    print('{}=={}'.format(o[0], o[-1]))
