import json
from html import escape
from .eval_utils import TabUnit, TabUnit_eval

def trope(text):
    return text.replace(',', '，').replace(';', '；').replace(':', '：') \
               .replace('?', '？').replace('!', '！').replace('(', '（') \
               .replace(')', '）').replace('[', '【').replace(']', '】') \
               .replace('"', '”').replace("'", '’')

def json2html(json_data, with_text=True, with_attr=False):
    new_html = "<html><body>"
    tables = json_data["tables"]
    # print(tables)
    for table in tables:
        new_html += "<table style=\"border:1px solid black;border-collapse:collapse\">"
        #table body
        new_html += "<tbody>"
        ySize = len(table)-2
        xSize = 0
        for i in range(1,len(table)-1):
            if(len(table[i])>xSize):
                xSize = len(table[i])
        #print('xSize %d, ySize %d'%(xSize, ySize))
        cellisNull = [0 for _ in range(xSize*4)]
        for i in range(1,len(table)-1):
            line = table[i]
            start = 0
            new_html+="<tr>"
            for cell in line:
                #print('sx %d, ex %d'%(cell["sx"],cell["ex"]))
                if with_text:
                    cell_context = cell['text']
                else:
                    cell_context = "*"
                #for text in cell["text"]:
                #    cell_context+=text
                for mm in range(cell["sx"],cell["ex"]+1):
                    if mm>=0 and mm<len(cellisNull):
                        cellisNull[mm]+=cell["ey"]+1-cell["sy"]
                #补空格子
                while start < cell["sx"]:
                    if start>=0 and start<len(cellisNull):
                        if(cellisNull[start]<=cell["sy"]):
                            if with_attr:
                                new_html+="<td style=\"border:1px solid black;border-collapse:collapse\" colspan=1 rowspan=1></td>"
                            else:
                                new_html+="<td></td>"
                            cellisNull[start]+=1
                            pass
                    start+=1
                #add
                if with_attr:
                    new_html+="<td style=\"border:1px solid black;border-collapse:collapse\" colspan=%d rowspan=%d>%s</td>"%(cell["ex"]+1-cell["sx"],cell["ey"]+1-cell["sy"], cell_context)
                else:
                    new_html+="<td>%s</td>"%(cell_context)
                start+=cell["ex"]+1-cell["sx"]
            new_html+="</tr>"
        new_html += "</tbody>"
        new_html += "</table>"
    new_html += "</body></html>"

    return new_html

def json2html2(json_data, with_text=True, with_attr=False):
    new_html = "<tabular>"
    tables = json_data["tables"]
    # print(tables)
    for table in tables:
        new_html += "<tbody>"
        ySize = len(table)-2
        xSize = 0
        for i in range(1,len(table)-1):
            if(len(table[i])>xSize):
                xSize = len(table[i])
        #print('xSize %d, ySize %d'%(xSize, ySize))
        cellisNull = [0 for _ in range(xSize*4)]
        for i in range(1,len(table)-1):
            line = table[i]
            start = 0
            new_html+="<tr>"
            for cell in line:
                #print('sx %d, ex %d'%(cell["sx"],cell["ex"]))
                if with_text:
                    cell_context = cell['text']
                else:
                    cell_context = "*"
                #for text in cell["text"]:
                #    cell_context+=text
                for mm in range(cell["sx"],cell["ex"]+1):
                    if mm>=0 and mm<len(cellisNull):
                        cellisNull[mm]+=cell["ey"]+1-cell["sy"]
                #补空格子
                while start < cell["sx"]:
                    if start>=0 and start<len(cellisNull):
                        if(cellisNull[start]<=cell["sy"]):
                            if with_attr:
                                new_html+="<tdy>"
                            else:
                                new_html+="<tdy>"
                            cellisNull[start]+=1
                            pass
                    start+=1
                #add
                if with_attr:
                    new_html+="<tdy>"
                else:
                    new_html+="<tdy>"
                start+=cell["ex"]+1-cell["sx"]
            new_html+="</tr>"
        new_html += "</tbody></tabular>"
    

    return new_html

def json2html_teds(json_data, with_text=True):
    new_html = "<html><body>"
    tables = json_data["tables"]
    # print(tables)
    for table in tables:
        new_html += "<table>"
        ySize = len(table)-2
        xSize = 0
        for i in range(1,len(table)-1):
            if(len(table[i])>xSize):
                xSize = len(table[i])
        #print('xSize %d, ySize %d'%(xSize, ySize))
        cellisNull = [0 for _ in range(xSize*4)]
        for i in range(1,len(table)-1):
            line = table[i]
            start = 0
            new_html+="<tr>"
            for cell in line:
                #print('sx %d, ex %d'%(cell["sx"],cell["ex"]))
                if with_text:
                    cell_context = cell['text']
                else:
                    cell_context = "*"
                #for text in cell["text"]:
                #    cell_context+=text
                for mm in range(cell["sx"],cell["ex"]+1):
                    if mm>=0 and mm<len(cellisNull):
                        cellisNull[mm]+=cell["ey"]+1-cell["sy"]
                #补空格子
                while start < cell["sx"]:
                    if start>=0 and start<len(cellisNull):
                        if(cellisNull[start]<=cell["sy"]):
                            new_html+="<td></td>"
                            cellisNull[start]+=1
                            pass
                    start+=1
                #add
                new_html+="<td>%s</td>"%(cell_context)
                start+=cell["ex"]+1-cell["sx"]
            new_html+="</tr>"
        new_html += "</table>"
    new_html += "</body></html>"
    
    return new_html

def cells_row_convert(cells):
    d = {}
    d['tables'] = [[]]
    d['tables'][0].append({'text':None,'type':"head"})
    mmmm = {}
    #每一行做归类
    for cell in cells:
        if cell['ysc'] not in mmmm:
            mmmm[cell['ysc']]=[cell]
        else:
            mmmm[cell['ysc']].append(cell)
    for sx in mmmm:
        #行
        ddd = []
        for cell in mmmm[sx]:
            c = {}
            c['sy'] = cell['ysc']
            c['ey'] = cell['yec']
            c['sx'] = cell['xsc']
            c['ex'] = cell['xec']
            if 'score' in cell:
                c['score'] = cell['score']
            else:
                c['score'] = 1
            if 'text' in cell:
                c['text'] = trope(cell['text'])
            else:
                c['text'] = ''
            ddd.append(c)
        d['tables'][0].append(ddd)
    d['tables'][0].append({'text':None,'type':"tail"})
    return d

def cells_col_convert(cells):
    d = {}
    d['tables'] = [[]]
    d['tables'][0].append({'text':None,'type':"head"})
    mmmm = {}
    #每一列做归类
    for cell in cells:
        if cell['xsc'] not in mmmm:
            mmmm[cell['xsc']]=[cell]
        else:
            mmmm[cell['xsc']].append(cell)
    for sx in mmmm:
        #列
        ddd = []
        for cell in mmmm[sx]:
            c = {}
            c['sy'] = cell['xsc']
            c['ey'] = cell['xec']
            c['sx'] = cell['ysc']
            c['ex'] = cell['yec']
            if 'text' in cell:
                c['text'] = trope(cell['text'])
            else:
                c['text'] = ''
            ddd.append(c)
        d['tables'][0].append(ddd)
    d['tables'][0].append({'text':None,'type':"tail"})
    return d

def is_priori(unit_a, unit_b):
  if unit_a.top_idx < unit_b.top_idx :
      return True
  elif unit_a.top_idx > unit_b.top_idx :
      return False
  if unit_a.left_idx < unit_b.left_idx :
      return True
  elif unit_a.left_idx > unit_b.left_idx :
      return False
  if unit_a.bottom_idx < unit_b.bottom_idx :
      return True
  elif unit_a.bottom_idx > unit_b.bottom_idx :
      return False
  if unit_a.right_idx < unit_b.right_idx :
      return True
  elif unit_a.right_idx > unit_b.right_idx :
      return False
  
def is_same_logi(unit_a, unit_b):
    if unit_a.top_idx == unit_b.top_idx and unit_a.bottom_idx == unit_b.bottom_idx and unit_a.left_idx == unit_b.left_idx and unit_a.right_idx == unit_b.right_idx:
        return True
    else:
        return False

def res2html(spatial, logi, cls_cell, score, text):
    #  把模型输出的logic-ax格式结果转为html格式，这里采用单张图的转换
    # result : dict, key表示目标类别, value: [K, 9]
    # spatial: [num_valid, 8] ndarray
    # logi: [num_valid, 4] ndarray
    # cls_cell: [num_valid] ndarray
    # score: [num_valid] ndarray
    # text: [num_valid] list
    ulist = []
    for m in range(len(spatial)):
      bbox = spatial[m]
      unit = TabUnit(bbox[:8], logi[m,:], cls_cell[m], score[m], text[m])
      ulist.append(unit)
    
    length = len(ulist)
    for index in range(length):
      for j in range(1, length-index):
        if is_priori(ulist[j], ulist[j-1]):
          ulist[j-1], ulist[j] = ulist[j], ulist[j-1]

    dup_indices = []
    for index in range(1, length):
        if is_same_logi(ulist[index], ulist[index-1]):
            ulist[index-1].text += ulist[index].text
            dup_indices.append(index)
    ulist_dup = [ulist[i] for i in range(length) if i not in dup_indices]
    ulist = ulist_dup

    cells = []
    for unit in ulist:
        cells.append({'ysc':unit.axis[0], 'yec': unit.axis[1], 'xsc': unit.axis[2], 'xec': unit.axis[3], 
                      'text': unit.text, 'cls': unit.cls_cell, 'score': unit.score})
    pred_row_json = cells_row_convert(cells)
    pred_row_html = json2html(pred_row_json, with_attr=True)
    # print(cells)
    # print(pred_row_json)
    return pred_row_html

def res2html_teds(spatial, logi, cls_cell):
    #  把模型输出的logic-ax格式结果转为适合teds的html格式，这里采用单张图的转换
    # result : dict, key表示目标类别, value: [K, 9]
    # spatial: [num_valid, 8] ndarray
    # logi: [num_valid, 4] ndarray
    # cls_cell: [num_valid] ndarray
    ulist = []
    for m in range(len(spatial)):
      bbox = spatial[m]
      unit = TabUnit_eval(bbox[:8], logi[m,:], cls_cell[m])
      ulist.append(unit)
    
    length = len(ulist)
    for index in range(length):
      for j in range(1, length-index):
        if is_priori(ulist[j], ulist[j-1]):
          ulist[j-1], ulist[j] = ulist[j], ulist[j-1]

    dup_indices = []
    for index in range(1, length):
        if is_same_logi(ulist[index], ulist[index-1]):
            # ulist[index-1].text += ulist[index].text
            dup_indices.append(index)
    ulist_dup = [ulist[i] for i in range(length) if i not in dup_indices]
    ulist = ulist_dup

    cells = []
    for unit in ulist:
        cells.append({'ysc':unit.axis[0], 'yec': unit.axis[1], 'xsc': unit.axis[2], 'xec': unit.axis[3], 
                      'cls': unit.cls_cell})
    pred_row_json = cells_row_convert(cells)
    pred_row_html = json2html_teds(pred_row_json)
    # print(cells)
    # print(pred_row_json)
    return pred_row_html

def res2json(spatial, logi, cls_cell, score, text):
    # result : dict, key表示目标类别, value: [K, 9]
    # spatial: [num_valid, 8] ndarray
    # logi: [num_valid, 4] ndarray
    # cls_cell: [num_valid] ndarray
    # score: [num_valid] ndarray
    # text: [num_valid] list

    # record the size of table
    max_col, max_row = -1, -1
    ulist = []
    for m in range(len(spatial)):
      bbox = spatial[m]
      unit = TabUnit(bbox[:8], logi[m,:], cls_cell[m], score[m], text[m])
      ulist.append(unit)
      if logi[m, 1] > max_row:
          max_row = logi[m, 1]
      if logi[m, 3] > max_col:
          max_col = logi[m, 3]
    
    num_rows = max_row + 1
    num_cols = max_col + 1
    
    length = len(ulist)
    for index in range(length):
      for j in range(1, length-index):
        if is_priori(ulist[j], ulist[j-1]):
          ulist[j-1], ulist[j] = ulist[j], ulist[j-1]

    dup_indices = []
    for index in range(1, length):
        if is_same_logi(ulist[index], ulist[index-1]):
            ulist[index-1].text += ulist[index].text
            dup_indices.append(index)
    ulist_dup = [ulist[i] for i in range(length) if i not in dup_indices]
    ulist = ulist_dup

    cells = []
    for unit in ulist:
        cell = {}
        cell['bbox'] = [unit.bbox.point1[0][0], unit.bbox.point1[0][1], unit.bbox.point2[0][0], unit.bbox.point2[0][1], 
                        unit.bbox.point3[0][0], unit.bbox.point3[0][1], unit.bbox.point4[0][0], unit.bbox.point4[0][1]]
        cell['logi'] = [unit.axis[0], unit.axis[1], unit.axis[2], unit.axis[3]]
        cell['cls'] = unit.cls_cell
        cell['score'] = unit.score
        cell['text'] = unit.text
        cells.append(cell)
    return cells, num_rows, num_cols

def format_html(tab):
    """
    Formats HTML code from tokenized annotation of table img
    """
    html_code = tab['html']['structure']['tokens'].copy()
    to_insert = [i for i, tag in enumerate(html_code) if tag in ('<td>', '>')]
    for i, cell in zip(to_insert[::-1], tab['html']['cells'][::-1]):
        if cell['tokens']:
            cell = [escape(token) if len(token) == 1 else token for token in cell['tokens']]
            cell = ''.join(cell)
            html_code.insert(i + 1, cell)
    html_code = ''.join(html_code)
    html_code = '''<html><body><table>%s</table></body></html>''' % html_code
    return html_code