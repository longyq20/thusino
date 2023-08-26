import numpy as np
import random
import pickle
from multiprocessing import Process
from multiprocessing import SimpleQueue
from datetime import datetime
import eva
from rlogging import recocdvalue

def load_dict(file_path):
    with open(file_path, 'rb') as file:
        dictionary = pickle.load(file)
    return dictionary

def savesolution(pop_num,rho,beta,num_exchange,num_mutation,num_to_mutate,trained_gene,par_pop,bestfitvalue):
    tt = datetime.now()
    strtt = tt.strftime("%Y.%m.%d_%H.%M")
    this_solu = {
        "resolvetime":tt,
        "pop_num": pop_num,
        "rho":rho,
        "beta":beta,
        "num_exchange":num_exchange,
        "num_mutation":num_mutation,
        "num_to_mutate":num_to_mutate,
        "trained_gene":trained_gene,
        "par_pop":par_pop,
        "bestfitvalue":bestfitvalue
        } 
    file_path = r"solutions/{}-best_fit {:.5f}.pickle".format(strtt,bestfitvalue)
    with open(file_path, 'wb') as file:
        pickle.dump(this_solu, file)

def loadsolution(file_path):
    resolu = {}
    with open(file_path, 'rb') as file:
        resolu = pickle.load(file)
    
    pop_num = resolu["pop_num"]
    rho = resolu["rho"]
    beta = resolu["beta"]
    num_exchange = resolu["num_exchange"]
    num_mutation = resolu["num_mutation"]
    num_to_mutate = resolu["num_to_mutate"]
    trained_gene = resolu["trained_gene"]
    par_pop = resolu["par_pop"]
    resolvetime = resolu["resolvetime"]
    #bestfitvalue = resolu["bestfitvalue"]
    
    print("从文件加载已有解...")
    for key,value in resolu.items():
        if key != "par_pop":
            print(key,":",value)
    return  pop_num,        \
            rho,            \
            beta,           \
            num_exchange,   \
            num_mutation,   \
            num_to_mutate,  \
            trained_gene,   \
            par_pop
        
def geninit(P, Q, L, pop_num,indishape,I,numB,numC):
    # 单纯考虑距离解
    # x-y-z,x-z-y,y-x-z,y-z-x,z-x-y,z-y-x
    # 初始解使用ndarray,三维pql分别表示一个位置
    pop = []
    for i in range(pop_num - 6):
        idv = I.copy()
        idv += [-i for i in range(1, 1097)]
        random.shuffle(idv)
        pop.append(np.reshape(idv, indishape))
    # pop.append(np.reshape(idv,indishape,order="A"))
    # pop.append(np.reshape(idv,indishape,order="F"))
    emut = [ (P, L, Q), (L, P, Q), (L, Q, P)]
    emutnum = [ (0, 2, 1), (1, 2, 0), (2, 1, 0)]

    roadwayB = [3,5,7,8,10]
    roadwayC = [1,4,6,9]
    ii = I.copy() + [-i for i in range(1, 1097)]
    for i in range(len(emut)): #将B按顺序添加到其中
        countb = 0
        countc = numB
        countn = numB + numC
        idv = np.zeros(indishape, dtype=int)
        for a in emut[i][0]:
            for b in emut[i][1]:
                for c in emut[i][2]:
                    emuabc = [a, b, c]
                    if emuabc[emutnum[i][0]] in roadwayB: #在B所在的分区
                        if countb >= numB:
                            idv[emuabc[emutnum[i][0]]][emuabc[emutnum[i][1]]][emuabc[emutnum[i][2]]] = ii[countn]
                            countn += 1
                        idv[emuabc[emutnum[i][0]]][emuabc[emutnum[i][1]]][emuabc[emutnum[i][2]]] = ii[countb]
                        countb += 1
                    if emuabc[emutnum[i][0]] in roadwayC: #在C所在的分区
                        if (countc - numB)>= numC:
                            idv[emuabc[emutnum[i][0]]][emuabc[emutnum[i][1]]][emuabc[emutnum[i][2]]] = ii[countn]
                            countn += 1
                        idv[emuabc[emutnum[i][0]]][emuabc[emutnum[i][1]]][emuabc[emutnum[i][2]]] = ii[countc]
                        countc += 1
        pop.append(idv.copy())
        
    emut = [ (Q, L, P), (L, P, Q), (L, Q, P)]
    emutnum = [ (2, 0, 1), (1, 2, 0), (2, 1, 0)]
    for i in range(len(emut)): #将B按顺序添加到其中
        count = 0
        idv = np.zeros(indishape, dtype=int)
        for a in emut[i][0]:
            for b in emut[i][1]:
                for c in emut[i][2]:
                    emuabc = [a, b, c]
                    idv[emuabc[emutnum[i][0]]][emuabc[emutnum[i][1]]][emuabc[emutnum[i][2]]] = ii[count]
                    count += 1
                    
        pop.append(idv.copy())
    return pop

def getp(parents_fit: np.ndarray):  # 获取轮盘赌概率
    #newfit = [(i-1000)/100 for i in parents_fit]
    newfit = parents_fit
    domi = sum(newfit)
    rever_prob = [i / domi for i in newfit]  # 数值越大的概率越大
    prob = [1 / i for i in rever_prob]
    factor = 1 / sum(prob)
    #print(newfit)
    ret = [i * factor for i in prob]
    #print(ret)
    return ret

def getp1(parents_fit: np.ndarray):  # 获取轮盘赌概率
    #newfit = [(i-1000)/100 for i in parents_fit]
    newfit  = []
    maxf = max(parents_fit)
    R = maxf - min(parents_fit)
    if R < 10 **-8:
        newfit = [1 for i in range(len(parents_fit))]
    else:
        for fit in parents_fit:
            newfit.append( 1+2*(maxf-fit)/R)
    domi = sum(newfit)
    ret = [i / domi for i in newfit]  # 数值越大的概率越大
    print(ret)
    return ret

def cross(num_exchage: int, pop: list, proulette: list,numall,indishape) -> list:  # 随机交换几个位置
    pop_num = len(pop)
    children = []
    full_index = np.arange(numall)  # 获取全部索引
    while len(children) < pop_num:  # 如果子代没有达到群体数，则继续
        parents = np.random.choice(np.arange(pop_num), 2, p=proulette, replace=False)  # 随机选择两个父母 || 按照轮盘赌选择两个父母
        prt1 = pop[parents[0]].flatten()
        prt2 = pop[parents[1]].flatten()
        cross_index = np.random.choice(full_index, num_exchage, replace=False)
        elemt_prt1 = prt1[cross_index]  # 找出父辈1中索引对应的元素
        elemt_prt2 = prt2[cross_index]  # 找出父辈2中索引对应的元素，将其方法index_part对应的位置上去

        chd2 = prt2.copy()
        for i, x in enumerate(elemt_prt1):
            id_p2 = np.where(chd2 == x)
            ele_p2 = chd2[cross_index[i]]
            chd2[id_p2] = ele_p2
            chd2[cross_index[i]] = x

        chd1 = prt1.copy()
        for i, x in enumerate(elemt_prt2):
            id_p1 = np.where(chd1 == x)
            ele_p1 = chd1[cross_index[i]]
            chd1[id_p1] = ele_p1
            chd1[cross_index[i]] = x

        children.append(np.reshape(chd1, indishape))  # 生成了1个孩子
    
    return children

def muatate(num_muatation: int, idi: np.ndarray,numall,indishape):  # 随机交换自己的元素
    nidi = idi.flatten()
    mutate_index = np.random.choice(np.arange(numall), num_muatation, replace=False)
    muatate_value = nidi[mutate_index]
    np.random.shuffle(muatate_value)
    for i, x in enumerate(mutate_index):
        nidi[x] = muatate_value[i]
    return np.reshape(nidi, indishape)

def elit(allindiv, indiv_evaluation, pop_num):  # 只保留2个精英,其余由子代排序代替
    parents = allindiv[:pop_num]
    children = allindiv[pop_num:]
    parents_evaluation = indiv_evaluation[:pop_num]
    children_evaluation = indiv_evaluation[pop_num:]
    p_index = np.argsort(parents_evaluation)
    c_index = np.argsort(children_evaluation)
    
    new_pop = parents[p_index[0]] #保留1个精英
    new_evaluation = parents_evaluation[p_index[0]]
    for i in range(pop_num-1):
        new_indiv = children[c_index[i]]
        new_pop.append(new_indiv)
        new_evaluation.append(children_evaluation[c_index[i]])

    best_fit = np.min(new_evaluation)
    return new_pop, best_fit, new_evaluation

def elit1(all_indiv:np.ndarray, indiv_evaluation,pop_num): #保留精英
    num_to_leave =int(pop_num//4)
    index = np.argsort(indiv_evaluation)
    new_pop = []
    new_evaluation = []
    for i in range(pop_num-num_to_leave):
        new_indiv = all_indiv[index[i]]
        new_pop.append(new_indiv) 
        new_evaluation.append(indiv_evaluation[index[i]])
    
    for j in range(num_to_leave):
        new_evaluation.append(indiv_evaluation[index[pop_num+j]]) #保留一个较差的个体
        new_pop.append( all_indiv[index[pop_num+j]])

    best_fit = indiv_evaluation[index[0]]
    return new_pop,best_fit,new_evaluation
    

def main(generation, pop_num=16, rho=10** -2, beta=10 ** -4, num_exchange=120, num_mutation=10,num_to_mutate=8):
    # 货位总数3780个
    # BC类共有料箱2684个，其中B类1677个，C类1007个
    num_process = 4 #种群的大小只能是线程数的整数倍
    hist_best =  42.5 #历史最优分数
    print("使用进程数 {}".format(num_process))
    tt = datetime.now()
    strtt = tt.strftime("%Y.%m.%d_%H.%M.%S")
    numB = 1677
    numC = 1007
    numall = 3780
    I = [i for i in range(numB + numC)]
    P = [i for i in range(10)]  # 行数
    Q = [i for i in range(21)]  # 列数
    L = [i for i in range(18)]  # 层数
    indishape = (10, 21, 18)

    dij = np.load("dij.npy")
    Dpql = np.load("Dpql.npy")
    boxindex2freq = {}
    boxindex2material = {}
    findex2cindex = {}
    boxindex2freq = load_dict("index2freq.pickle")
    boxindex2material = load_dict("index2mate.pickle")
    findex2cindex = load_dict("findex2cindex.pickle")
    materialrelation =load_dict("ralationdict.pickle")
    
    par_pop = []
    trained_gene = 0
    load_flag = True
    if  load_flag: #如果从文件加载
        file_path = r"solutions/2023.08.26_22.00-best_fit 42.54971.pickle"
        if file_path == "":
            print("文件名为空或不存在!")
            return 0;
        else:
            pop_num,        \
            rho,            \
            beta,           \
            num_exchange,   \
            num_mutation,   \
            num_to_mutate,  \
            trained_gene,   \
            par_pop = loadsolution(file_path)
    else:
        par_pop = geninit(P,Q,L,pop_num,indishape,I,numB,numC) 
        #par_pop = list(np.load(r"solutions\2023.08.26_09-32-poplation-gene50.npy"))
        print("模型初始化...")
        print("种群代数{}, 种群大小{}, 交换个数{}, 变异个数{}, 变异个体数{}".format(generation,pop_num,num_exchange,num_mutation,num_to_mutate))
    
    num_to_mutate = 8 #f强制覆写变异量
    raw_fit = []
    queues = [ SimpleQueue() for i in range(num_process)]
    jobs = []
    print("计算初始种群适应度...")
    for i in range(num_process):
        #p = Process(target=evaluate4,args=(par_pop[int(pop_num/4*i):int(pop_num/4*(i+1))], rho, beta,boxindex2freq,findex2cindex,Dpql,boxindex2material,materialrelation,dij,q,i,))
        p = Process(target=eva.evaluate,args=(par_pop, rho, beta,boxindex2freq,findex2cindex,Dpql,boxindex2material,materialrelation,dij,queues[i],i,num_process,))
        p.start()
        jobs.append(p)
    for p in jobs:
        p.join()
    for q in queues:
        for _qi in range(int(pop_num/num_process)):
            raw_fit.append(q.get())
        q.close()
    parents_evaluation = [i[0]+i[1] for i in raw_fit]
    
    for _gene in range(generation):
        # print("第{}代适应度: ".format(_),parents_evaluation)
        print("第{}代适应度>>".format(_gene))
        for item in parents_evaluation:
            print("{:.5f}".format(item), end=" ")
        print("")  # 换行
        
        print("交叉互换...")
        p = getp1(parents_evaluation)  # 轮盘赌概率
        child_pop = cross(num_exchange, par_pop, p,numall,indishape)  # 交叉互换

        print("变异...")
        index_to_mutate = np.random.choice(pop_num, num_to_mutate, replace=False)  # 发生编译的个体索引
        for i in index_to_mutate:
            child_pop.append(muatate(num_mutation, child_pop[i],numall,indishape))  # 变异后的个体添加到子代中

        print("计算子代适应度...")
        raw_fit = [] 
        queues = [SimpleQueue() for i in range(num_process) ]
        jobs = []
        for i in range(num_process):
            #p = Process(target=evaluate,args=(child_pop[int((pop_num+num_to_mutate)/4*i):int((pop_num+num_to_mutate)/4*(i+1))], rho, beta,boxindex2freq,findex2cindex,Dpql,boxindex2material,materialrelation,dij,q,i,))
            p = Process(target=eva.evaluate,args=(child_pop, rho, beta,boxindex2freq,findex2cindex,Dpql,boxindex2material,materialrelation,dij,queues[i],i,num_process,))
            p.start()
            jobs.append(p)
        for p in jobs:
            p.join()
        for q in queues:
            for _qi in range(int((pop_num+num_to_mutate)/num_process)):
                raw_fit.append(q.get())
            q.close()

        child_evalation = [i[0]+i[1] for i in raw_fit]
        all_pop = par_pop + child_pop  # 总的群体
        all_evaluation = parents_evaluation + child_evalation

        print("保留精英个体...")
        par_pop, best_fit, parents_evaluation = elit(all_pop, all_evaluation, pop_num)

        print("第{}代最优个体分数: {}".format(_gene + 1, best_fit))

        if _gene % 50 == 0 or (hist_best - best_fit)>0.05 :  # 每隔10代进行一次存档
            if (hist_best - best_fit) > 0.5:
                hist_best = best_fit
            print("第{}代最佳个体已经存档...".format(_gene))
            recocdvalue(best_fit,_gene,strtt)
            savesolution(pop_num, rho, beta, num_exchange, num_mutation,num_to_mutate,_gene+trained_gene,par_pop,best_fit)
            
    np.save("bestindiv-{:.5f}.npy".format(best_fit), par_pop[0])

if __name__ == "__main__":
    main(800, 16)



