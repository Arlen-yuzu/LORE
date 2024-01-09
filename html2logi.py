import jsonlines
from tqdm import tqdm
import numpy as np


class Map():
    def __init__(self, size = 100):
        self.size = size
        self.maps = np.zeros((size, size+1))
        self.row_pointer = 0
        self.col_pointer = 0
        self.axis_list = []
        self.is_complex = 0
        
    def another_row(self):
        self.row_pointer = self.row_pointer + 1
        self.col_pointer = int(np.argwhere(self.maps[self.row_pointer] == 0)[0])
    
    def set_col(self):
        # set the col_pointer to the first empty unit
        if self.maps[self.row_pointer][self.col_pointer] != 0:
            for i in range(self.size):
                if self.maps[self.row_pointer][self.col_pointer] != 0:
                    self.col_pointer = self.col_pointer + 1
                else:
                    break
    
    def add_unit(self, colspan, rowspan):
        if colspan + rowspan > 2:
            self.is_complex += 1
        
        axis = [self.row_pointer]
        for i in range(colspan):
            for j in range(rowspan):
                self.maps[self.row_pointer + j][self.col_pointer + i] += 1
        
        axis.append(self.row_pointer + rowspan - 1)
        axis.append(self.col_pointer)
        axis.append(self.col_pointer + colspan - 1)
        
        self.axis_list.append(axis)
        
        self.col_pointer = self.col_pointer + colspan
        # 需要对列指针更新后的位置是否被前面几行的跨行单元占用进行检查，从而更新列指针位置
        self.set_col()
        
    def shape(self):
        width = int(np.argwhere(self.maps[0] != 0)[-1])
        height = int(np.argwhere(np.transpose(self.maps)[0] != 0)[-1])
        return width + 1, height + 1
    
    def check(self):
        width, height = self.shape()
        if self.maps.max() > 1:
            #print('ERROR: Overlapping units.')
            return False
        elif int(width * height) != int(self.maps.sum()):
            #print('ERROR: Lossing units.')
            return False
        elif len(self.axis_list) != len(self.cls_list):
            print('ERROR: Lossing class information.')
            return False
        else:
            return True
        
    def count(self):
        return len(self.axis_list)
    
    def ncomplex(self):
        return self.is_complex

def process_table(tab):    
    bgr = 0 # beginning of row
    bgd = 0 # beginning of unit
    bgsd = 0 # beginning of spanning unit

    colspan = 1
    rowspan = 1

    new_spaning = 0
    init = 0

    maps = Map()

    for token in tab['html']['structure']['tokens']:
        if token == '>' or token == '</tbody>':
            continue
        elif token == '<thead>':
            continue
        elif token == '</thead>':
            continue
        elif token == '<tbody>':
            continue
        elif token == '<tr>':
            if bgr == 0:
                bgr = 1
            else:
                print('ERROR: Starting a new row without an endding of the former row.')
                quit()

            if init == 0:
                init = init + 1
            else:
                maps.another_row()

        elif token == '</tr>':
            if bgr == 1:
                bgr = 0
            else:
                print('ERROR: Endding a non-existing row.')
                quit()

        elif token == '<td>':
            if bgd == 0:
                bgd = 1
            else:
                print('ERROR: Starting a new unit without an endding of the former unit.')
                quit()

            maps.add_unit(colspan = 1, rowspan = 1)

        elif token == '<td':
            if bgsd == 0:
                bgsd = 1
            else:
                print('ERROR: Starting a new spanning nit without an endding of the former spanning unit.')
                quit()

        elif 'colspan' in token:
            colspan = int(token.split('=')[1].strip('"'))

        elif 'rowspan' in token:
            rowspan = int(token.split('=')[1].strip('"'))

        elif token == '</td>':
            if bgsd == 1:
                maps.add_unit(colspan = colspan, rowspan = rowspan)
            if bgd + bgsd == 1:
                bgd = bgd - bgd
                bgsd = bgsd - bgsd
                rowspan = 1
                colspan = 1
            else :
                print('ERROR: Endding a non-existing unit.')
                quit()            
        else:
            print('ERROE: Wrong Tag: {}'.format(token))
            quit()
    if not maps.checks():
        print('ERROR: Wrong table.')
        quit()
    return maps


with jsonlines.open('PubTabNet_2.0.0.jsonl') as reader:
    for obj_ind, obj in tqdm(enumerate(reader)):
        tab = obj
        maps = process_table(tab)
        print(maps.axis_list)
