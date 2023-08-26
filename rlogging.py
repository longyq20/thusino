from datetime import datetime

def recocdvalue(bestfit,_gene,timestr):
    file_path = r"solutions/{}-log.txt".format(timestr)
    file_path
    tt = datetime.now()
    strtt = tt.strftime("%Y/%m/%d %H:%M:%S")
    mess = strtt + "    iteration: {}    best_fitness: ".format(_gene)+ str(bestfit)
    
    with open(file_path,"a") as file:
        file.write(mess)
        file.write("\n")