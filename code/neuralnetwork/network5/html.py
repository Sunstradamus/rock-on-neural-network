
def getHead():
    return '''<!DOCTYPE html>
<html>

<head>
<title>HTML Tables</title>
</head>

<body>
  <table border="1">
    <tr>
      <th>image</th>
      <th>Paper</th>
      <th>Rock</th>
      <th>Scissors</th>
    </tr>
'''

def getTail(correct,total):
    string =  "  </table>\n"
    string += "Score: %d/%d" %(correct, total)
    string += "</body>\n</html>"
    return string

def makeCell(text, bgcolor=""):
    if bgcolor == "":
        return '      <td>%s</td>\n' % (text)
    else:
        return '      <td bgcolor="%s">%s</td>\n' % (bgcolor, text)

def makeRow(image, values, correct, wrong):
    image = image.replace(" ", "%20")
    image = image.replace("#", "%23")
    string = '''    <tr>\n      <td>'''
    string += '<img src=\"%s\" style="width:299px;height:299px;">' % (image)
    string += "</td>" + "\n"

    for i in range(3):
        if correct == i:
            string += makeCell(values[i], "00FF00")
        elif wrong == i:
            string += makeCell(values[i], "FF0000")
        else:
            string += makeCell(values[i])
    string += "    </tr>\n"
    return string
