import fnmatch
import os
import pandas as pd
import numpy as np
import re

file1 = open("formulas2.lst", "r",encoding='UTF-8')
file_list1 = file1.readlines()
file_list=[]
latex_list = []
math_list = []
j = 0
latex_len = file_list1.__len__()
os.chdir('C:\work\ml_PDF_parsing\ML_MathToLatex\dataset\images')
images = 'C:\work\ml_PDF_parsing\ML_MathToLatex\dataset\images' 
images_list = os.listdir(images)

print(latex_len)
print(len(images_list))
for i in range(latex_len):
    image_name = images_list[i] 
    a=str(file_list1[i])
    # print(f"the string is {a}")
    if(a.__contains__('\label{')):
        pattern = r'\\label{.*?\}'
        mod_string = re.sub(pattern,'',a)
        if(mod_string.endswith(',') or mod_string.endswith('.')):
            mod_string = mod_string[:-1]
        elif(mod_string.endswith(' .') or mod_string.endswith(' ,')):
            mod_string = mod_string[:-2]
        # print(f"mod string is {mod_string}")
        latex_list.append(mod_string)

    else:
        latex_list.append(a)
    math_list.append(image_name)

    # if j==100:
        # break
    # j+=1
# latex_list = pd.Series(latex_list)
data_tuples = list(zip(math_list,latex_list))
df = pd.DataFrame(data_tuples, columns=['image','latex'])
df.to_csv('C:\work\ml_PDF_parsing\ML_MathToLatex\dataset\data.csv',index=True)