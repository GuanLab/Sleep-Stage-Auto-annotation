#!/usr/bin/env python
import random
from datetime import datetime

random.seed(0)

path1 = './'
all_ids_name = 'ids_shhs1_1000.txt'
# train:test = 8:2
select_number = 800
selected_file_name = 'id_train80.txt'
else_file_name = 'id_test20.txt'

# read all lines
all_ids=open(path1 + all_ids_name,'r')
all_line=[]
for line in all_ids:
    all_line.append(line.rstrip())
all_ids.close()

# select lines

random.shuffle(all_line)
select_line=all_line[0:select_number]
else_line=all_line[select_number:]

# write selected lines
selected_ids=open(path1 + selected_file_name, 'w')
for line in select_line:
    selected_ids.write('%s' % line)
    selected_ids.write('\n')
selected_ids.close()
# write else lines
else_ids=open(path1 + else_file_name, 'w')
for line in else_line:
    else_ids.write('%s' % line)
    else_ids.write('\n')
else_ids.close()


# train:test = 1:1
select_number = 500
selected_file_name = 'id_train50.txt'
else_file_name = 'id_test50.txt'

# read all lines
all_ids=open(path1 + all_ids_name,'r')
all_line=[]
for line in all_ids:
    all_line.append(line.rstrip())
all_ids.close()

# select lines

random.shuffle(all_line)
select_line=all_line[0:select_number]
else_line=all_line[select_number:]

# write selected lines
selected_ids=open(path1 + selected_file_name, 'w')
for line in select_line:
    selected_ids.write('%s' % line)
    selected_ids.write('\n')
selected_ids.close()
# write else lines
else_ids=open(path1 + else_file_name, 'w')
for line in else_line:
    else_ids.write('%s' % line)
    else_ids.write('\n')
else_ids.close()
