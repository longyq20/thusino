from multiprocessing import SimpleQueue
import os

def evaluate4(par_pop: list, rho: float, beta,boxindex2freq:dict,findex2cindex:dict,Dpql:dict,boxindex2material:dict,materialrelation:dict,dij:dict,q:SimpleQueue,pid) :  #计算fitness
    #计算目标函数一：各个料箱到工作台的距离*出库频率
    pop_num =len(par_pop)
    sub_pop = par_pop[int(pop_num/4*pid):int(pop_num/4*(pid+1))]
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
            cindexi = findex2cindex[i]
            matenamei = "" if x < 0 else boxindex2material[x]
            for j in range(i+1,len(findiv)):
                y = findiv[j]
                cindexj = findex2cindex[j]
                matenamej =  "" if y < 0 else boxindex2material[y]
                rij = 0 if matenamei == "" or matenamej == "" else materialrelation[(matenamei,matenamej)]
                d_ij = rij* dij[cindexi+cindexj]
                Dis_ij += d_ij
        ret = (Dis_to_station*beta, Dis_ij*rho*beta)
        q.put(ret)
        print("pid:{:>5} idv:{:=2}->Di:{:=.5f},dij:{:=.5f}, sum: {:=.5f}".format(os.getpid(),pid*len(sub_pop) + (_ind+1),ret[0],ret[1],ret[0]+ret[1]),flush=True)
        #print("pid:{:>5} idv:{:=2}>>Di:{:=.5f},dij:{:=.5f}".format(os.getpid(),par_pop.index(indiv) + 1,ret[0],ret[1]),flush=True)
