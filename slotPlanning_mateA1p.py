
import numpy as np
import random
import pickle
from datetime import datetime
from rlogging import recocdvalue

def load_dict(file_path):
    with open(file_path, 'rb') as file:
        dictionary = pickle.load(file)
    return dictionary

def savesolution(pop_num,rho,beta,num_exchange,num_mutation,num_to_mutate,trained_gene,par_pop):
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
        "par_pop":par_pop
        } 
    file_path = r"solutions/{}_poplation.npy".format(strtt)
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


def geninit(P,Q,L,pop_num,indishape,I):
    #单纯考虑距离解
    #x-y-z,x-z-y,y-x-z,y-z-x,z-x-y,z-y-x
    #初始解使用ndarray,三维pql分别表示一个位置
    pop  = []
    for i in range(pop_num-6):
        idv = I.copy()
        idv += [-i for i in range(1,10)]
        random.shuffle(idv)
        pop.append(np.reshape(idv,indishape))
    #pop.append(np.reshape(idv,indishape,order="A"))
    #pop.append(np.reshape(idv,indishape,order="F"))
    emut = [(P,Q,L),(P,L,Q),(Q,P,L),(Q,L,P),(L,P,Q),(L,Q,P)]
    emutnum = [(0,1,2),(0,2,1),(1,0,2),(2,0,1),(1,2,0),(2,1,0)]
    ii =  I.copy() +  [-i for i in range(1,10)] 
    for i in range(len(emut)):
        count = 0 
        idv = np.zeros(indishape,dtype = int)
        for a in emut[i][0]:
            for b in emut[i][1]:
                for c in emut[i][2]: 
                    emuabc = [a,b,c]
                    idv[emuabc[emutnum[i][0]]][emuabc[emutnum[i][1]]][emuabc[emutnum[i][2]]] = ii[count]
                    count+=1

        pop.append(idv.copy())
    return pop


def getp(parents_fit:np.ndarray): #获取轮盘赌概率
    #newfit = [(i-30000)/1000 for i in parents_fit]
    newfit = parents_fit
    domi = sum(newfit)
    rever_prob = [ i/domi for i in newfit] #数值越大的概率越大
    prob = [1/i for i in rever_prob]
    factor = 1/sum(prob)
    ret = [i*factor for i in prob]
    #print(newfit)
    #print(ret)
    return ret

def getp1(parents_fit: np.ndarray):  # 获取轮盘赌概率
    #newfit = [(i-1000)/100 for i in parents_fit]
    newfit  = []
    maxf = max(parents_fit)
    R = maxf - min(parents_fit)
    if R < 10 **-10:
        newfit = [1 for i in range(len(parents_fit))]
    else:
        for fit in parents_fit:
            newfit.append( 1+(maxf-fit)/R)
    domi = sum(newfit)
    ret = [i / domi for i in newfit]  # 数值越大的概率越大
    print(ret)
    return ret

def cross(num_exchage:int,pop:list,proulette:list,indishape) -> list: #随机交换几个位置
    pop_num = len(pop)
    children = []
    full_index = np.arange(3240)  #获取全部索引
    while len(children) < pop_num: #如果子代没有达到群体数，则继续
        parents = np.random.choice(np.arange(pop_num),2,p = proulette ,replace=False) #随机选择两个父母 || 按照轮盘赌选择两个父母
        prt1 = pop[parents[0]].flatten()
        prt2 = pop[parents[1]].flatten()
        cross_index = np.random.choice(full_index,num_exchage,replace=False)
        elemt_prt1 = prt1[cross_index] #找出父辈1中索引对应的元素
        elemt_prt2 = prt2[cross_index] #找出父辈2中索引对应的元素，将其方法index_part对应的位置上去
        
        chd2 = prt2.copy()
        for i,x in enumerate(elemt_prt1):
            id_p2 = np.where(chd2 == x) 
            ele_p2 = chd2[cross_index[i]] 
            chd2[id_p2] = ele_p2
            chd2[cross_index[i]] = x
        
        chd1 = prt1.copy()
        for i,x in enumerate(elemt_prt2):
            id_p1 = np.where(chd1 == x) 
            ele_p1 = chd1[cross_index[i]] 
            chd1[id_p1] = ele_p1
            chd1[cross_index[i]] = x
        children.append(np.reshape(chd1,indishape)) #生成了1个孩子
    return children


def muatate(num_muatation:int,idi:np.ndarray,indishape): #随机交换自己的元素
    nidi = idi.flatten()
    mutate_index = np.random.choice(np.arange(3240),num_muatation,replace=False)
    muatate_value = nidi[mutate_index]
    np.random.shuffle(muatate_value)
    for i,x in enumerate(mutate_index):
        nidi[x]  = muatate_value[i]
    return np.reshape(nidi,indishape)


def evaluate1(sub_pop: list, rho: float, beta,boxindex2freq:dict,findex2cindex:dict,Dpql:dict,boxindex2material:dict,materialrelation:dict,dij:dict) :  #计算fitness
    #计算目标函数一：各个料箱到工作台的距离*出库频率
    retv = []
    for _ind,indiv in enumerate(sub_pop):
        findiv = indiv.flatten() 
        Dis_to_station = 0
        for i,x in enumerate(findiv): #迭代器
            freq =  0 if x<0 else boxindex2freq[x] # 料箱x对应的频率
            cindex = findex2cindex[i] # 在立体数组中的索引
            Dx = Dpql[cindex]
            Dis_to_station += Dx * freq
        #计算目标函数二：各个料箱之间的相关性*距离
        Dis_ij  = 0
        
        for i in range(len(findiv)-1):
            x = findiv[i]
            #freqi = 0 if x<0 else  boxindex2freq[x] 
            cindexi = findex2cindex[i]
            matenamei = "" if x < 0 else boxindex2material[x]
            for j in range(i+1,len(findiv)):
                y = findiv[j]
                #freqj = 0 if y<0 else boxindex2freq[y] 
                cindexj = findex2cindex[j]
                matenamej =  "" if y < 0 else boxindex2material[y]
                rij = 0 if matenamei == "" or matenamej == "" else materialrelation[(matenamei,matenamej)]
                d_ij = rij* dij[cindexi+cindexj]
                Dis_ij += d_ij
        ret = (beta * Dis_to_station,  Dis_ij * rho * beta) 
        retv.append((ret[0], ret[1]))
        print("idv:{:=2}>>Di:{:>.5f}, dij:{:>.5f}, sum:{:>.5f}".format((_ind+1),ret[0],ret[1],ret[0]+ret[1]),flush=True)
    return retv

def elit(all_indiv:np.ndarray, indiv_evaluation,pop_num): #保留精英
    index = np.argsort(indiv_evaluation)
    new_pop = []
    new_evaluation = []
    for i in range(pop_num):
        new_indiv = all_indiv[index[i]]
        new_pop.append(new_indiv) 
        new_evaluation.append(indiv_evaluation[index[i]])
    
    best_fit = indiv_evaluation[index[0]]
    return new_pop,best_fit,new_evaluation

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
    
    
def main(generation,pop_num = 16,rho=10**-3,beta = 10**-5,num_exchange = 120,num_mutation = 9,num_to_mutate = 4):
    tt = datetime.now()
    strtt = tt.strftime("%Y.%m.%d_%H.%M.%S")

    I = [i for i in range(3231)]
    P = [i for i in range(10)] # 行数
    Q = [i for i in range(18) ] # 列数
    L = [i for i in range(18)] #层数
    indishape = (10,18,18)

    dij = np.load("d_ij.npy")
    Dpql = np.load("Dpql.npy")
    boxindex2freq   = {}
    boxindex2material = {}
    findex2cindex = {}
    boxindex2freq = load_dict("boxindex2freq.pickle")
    boxindex2material = load_dict("boxindex2material.pickle")
    findex2cindex = load_dict("findex2cindex.pickle")
    materialrelation = load_dict("materialrelation.pickle")
    
    par_pop = []
    trained_gene = 0
    load_flag = False
    if  load_flag: #如果从文件加载
        file_path = ""
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
        par_pop = geninit(P,Q,L,pop_num,indishape,I) 
        #par_pop = list(np.load(r"solutions\2023.08.26_09-32-poplation-gene50.npy"))
        print("模型初始化...")
        print("种群代数{}, 种群大小{}, 交换个数{}, 变异个数{}, 变异个体数{}".format(generation,pop_num,num_exchange,num_mutation,num_to_mutate))
    
    print("计算初始种群适应度...")
    raw_fit = evaluate1(par_pop, rho, beta,boxindex2freq,findex2cindex,Dpql,boxindex2material,materialrelation,dij)
    parents_evaluation = [i[0]+i[1] for i in raw_fit]
    
    for _gene in range(generation):
        
        print("第{}代适应度>>".format(_gene))
        for item in parents_evaluation:
                print("{:.4f}".format(item),end=" ",flush=True)
        print("") #
        
        p = getp1(parents_evaluation) # 轮盘赌概率
        print("交叉互换>>>...")
        child_pop = cross(num_exchange,par_pop,p,indishape) # 交叉互换
        
        print("变异>>>...")
        index_to_mutate = np.random.choice(pop_num,num_to_mutate,replace=False) #发生编译的个体索引
        for i_to_mutate in index_to_mutate: 
            child_pop.append(muatate(num_mutation,child_pop[i_to_mutate],indishape)) #变异后的个体添加到子代中
        
        print("计算子代适应度...")
        raw_fit = evaluate1(child_pop, rho, beta,boxindex2freq,findex2cindex,Dpql,boxindex2material,materialrelation,dij)
        child_evalation = [i[0]+i[1] for i in raw_fit]
        all_pop = par_pop + child_pop #总的群体
        all_evaluation = parents_evaluation + child_evalation
        
        print("保留精英个体>>>...")
        par_pop,best_fit,parents_evaluation = elit1(all_pop,all_evaluation,pop_num)
        print("第{}代最优个体分数: {}".format(_gene+1,best_fit))
        
        if _gene % 50 == 0: #每隔10代进行一次存档
            print("第{}代最佳个体已经存档...".format(_gene))
            recocdvalue(best_fit,_gene,strtt)
            savesolution(pop_num, rho, beta, num_exchange, num_mutation,num_to_mutate,_gene+trained_gene,par_pop)
    np.save("bestindiv.npy",par_pop[0])

if __name__ == "__main__":
    main(20000,12)

