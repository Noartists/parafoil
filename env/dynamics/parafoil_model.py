# from main_doc_experiment import para
from env.dynamics.w2wx import w2wx
from env.dynamics.omega2angle_rate import omega2angle_rate
import math
import numpy as np


def parafoil_model(y,t, para):
    y = np.array([y]).T
    #print('执行一次了')
    # 调试
    # print("time:", t)
    # print("Shape of y:", y.shape)  # 打印 y 的形状
    # print("Value of y:", y)  # 打印 y 的值
    # print("维度",y.ndim)
    # print("大小", y.size)
    Ic = canopy_inertial(para)  # 伞体转动惯量，包含伞衣质量和伞内空气质量
    m_air = para.Rho * para.b * para.c * para.t / 2  # 伞内空气质量
    m_total = para.mc + m_air  # 总质量=伞体质量+伞内空气质量
    #Ic_r1 = np.concatenate(m_total * np.eye(3), np.zeros((3, 3)), axis=1)
    Ic_r1 = np.concatenate((m_total * np.eye(3), np.zeros((3, 3))), axis=1)
    Ic_r2 = np.concatenate((np.zeros((3, 3)), Ic), axis=1)
    Ic_r = np.concatenate((Ic_r1, Ic_r2), axis=0)
    # Ic_r = [m_total*eye(3),zeros(3,3);zeros(3,3),Ic]
    Ip = para.mp / 12 * 0.5 * np.eye(3)  # %负载转动惯量，Ip=m/12*diag([w2+h2,d2+h2,w2+d2])
    Ic_a = apparent_mass(para)  # 附加质量
    Bc_r = angle2dcm(0, para.miu, 0)  # 转换矩阵Tc-r，由伞体坐标系向安装坐标系转换

    # Ir = [Bc_r',zeros(3,3);zeros(3,3),Bc_r']*(Ic_r + Ic_a)*[Bc_r,zeros(3,3);zeros(3,3),Bc_r];
    Ir_1 = np.concatenate((Bc_r.T, np.zeros((3, 3))), axis=1)
    Ir_2 = np.concatenate((np.zeros((3, 3)), Bc_r.T), axis=1)
    Ir1 = np.concatenate((Ir_1, Ir_2), axis=0)
    Ir_3 = np.concatenate((Bc_r, np.zeros((3, 3))), axis=1)
    Ir_4 = np.concatenate((np.zeros((3, 3)), Bc_r), axis=1)
    Ir2 = np.concatenate((Ir_3, Ir_4), axis=0)
    Ir = np.dot(np.dot(Ir1, (Ic_r + Ic_a)), Ir2)

    A1 = Ir[:3, :3]
    A2 = Ir[:3, 3:6]
    A3 = Ir[3:6, :3]
    A4 = Ir[3:6, 3:6]

    angle = y[3:6]  # 伞体姿态角
    thetar = y[6][0]  # theta r 相对俯仰角
    psir = y[7][0]  # fai r 相对偏航角
    v0 = y[8:11]  # 伞体速度
    w = y[11:14]  # 伞体角速度
    ws = y[17:]  # 负载角速度
    # vp = Rbs*(v0 + w2wx(para.rcOc)*w) - w2wx(para.rcOp)*ws
    vp = y[14:17]  # 负载速度

    tmp1 = np.array([[-1, 0, -np.sin(thetar)],
                     [0, 1, 0],
                     [0, 0, np.cos(thetar)]])  # Eq.(22)

    tmp2 = np.linalg.solve(tmp1, np.array([[0], [ws[1][0]], [ws[2][0]]]) - np.dot(angle2dcm(psir, thetar, 0),
                                                                                  w))  # Eq.(23)
    # angle2dcm(psir,thetar,0)为Tc-p，由翼伞坐标系像负载坐标系转换
    # 这部分属于角速度约束，利用此式求出相对俯仰角导数

    dthetar = tmp2[1][0]  # dtheta r相对俯仰角导数、dfai r相对偏航角导数
    dpsir = tmp2[2]  # dfai r相对偏航角导数
    d_s = min(para.left, para.right)  # delta s对称挠度
    d_l = para.left  # 左边缘偏转控制
    d_r = para.right  # 右边缘偏转控制
    d_a = d_l - d_r  # delta a非对称挠度

    K81 = np.array([[np.cos(thetar), 0, np.sin(thetar)]])  # K1
    K82 = np.array([[np.cos(psir), np.sin(psir), 0]])  # K2

    mat1 = dthetar * np.array([[np.sin(thetar), 0, -np.cos(thetar)]])
    mat2 = dpsir * np.array([[-np.sin(psir), np.cos(psir), 0]])
    dps = np.dot(mat1, ws) + np.dot(mat2, w)  # 此时生成的dps1是维度为2，1*1的数组
    # dps = dthetar*[sin(thetar)    0     -cos(thetar)]*ws + dpsir*[-sin(psir)   cos(psir)   0]*w;%%Eq.(26)右半部分

    Rbs = angle2dcm(psir, thetar, 0)  # 转换矩阵Tc-p，伞体坐标系到负载坐标系
    Rnb = angle2dcm(angle[2][0], angle[1][0], angle[0][0])  # 转换矩阵Ti-c，惯性（大地）坐标系到伞体坐标系
    Fcg = np.dot(Rnb, np.array([[0, 0, para.mc * para.gn]]).T)  # 伞体坐标系中翼伞重力
    Fsg = np.dot(np.dot(Rbs, Rnb), np.array([[0, 0, para.mp * para.gn]]).T)  # 负载坐标系中负载重力
    E8 = np.array([[1], [0], [0]])  # 貌似没用
    # Mcz = [0 0 para.k_psi*psir+para.k_r*dpsir]';
    Mcz = para.k_r * dpsir  # 为Eq.(39)做铺垫
    mpf = - np.array([[0], [para.k_f * thetar + para.c_f * dthetar], [0]])
    mcf = - np.dot(Rbs.T, mpf)

    va = np.dot(Bc_r, v0 - np.dot(Rnb, para.vw))  # Va翼伞空速，va是3*1数组
    alpha = math.atan2(va[2][0] , va[0][0])
    va_2 = np.dot(va.T, va)
    beta = math.asin(va[1][0] / math.sqrt(va_2[0][0]))
    Bwp = angle2dcm(beta, alpha - math.pi, 0)  # 转换矩阵Tw-c，风场坐标系向伞体坐标系转换,但论文里是pi - alpha有不一致

    Faero_1 = para.Rho * para.As * np.dot(Bc_r.T, Bwp) * 0.5 * va_2[0][0]
    Faero_2_1 = para.CD0 + para.CDa2 * pow(alpha, 2) + para.CDds * d_s
    Faero_2_2 = para.CYbeta * beta
    Faero_2_3 = para.CL0 + para.CLa * alpha + para.CLds * d_s
    Faero_2 = np.array([[Faero_2_1, Faero_2_2, Faero_2_3]]).T
    Faero = np.dot(Faero_1, Faero_2)  # 翼伞气动力，Eq.(12)有不一致 3*1

    Maero_1 = Bc_r.T * para.As * para.Rho * 0.5 * va_2
    Maero_2_1 = para.b * (para.Clbeta * beta + para.b / (2 * math.sqrt(va_2[0][0])) * (
                para.Clp * w[0][0] + para.Clr * w[2][0]) + para.Clda * d_a)
    Maero_2_2 = para.c * (para.Cm0 + para.Cma * alpha + para.b / (2 * math.sqrt(va_2[0][0])) * para.Cmq * w[1][0])
    Maero_2_3 = para.b * (para.Cnbeta * beta + para.b / (2 * math.sqrt(va_2[0][0])) * (
                para.Cnp * w[0][0] + para.Cnr * w[2][0]) + para.Cnda * d_a)
    Maero_2 = np.array([[Maero_2_1, Maero_2_2, Maero_2_3]]).T
    Maero = np.dot(Maero_1, Maero_2)  # 翼伞气动力距，Eq.(13)有不一致 3*1

    vas = vp - np.dot(np.dot(Rbs, Rnb) , para.vw)
    vas_2 = vas.T * vas
    Fsa = - para.CDp * (0.5 * para.Rho * para.Ap * math.sqrt(vas_2[0][0])) * vas  # 回收物所受气动力

    e_t = np.array([[1, 0], [0, 0], [0, 1]])  # 论文中的E1+E2

    E1 = np.concatenate((A1, A2, np.zeros((3, 3)), np.zeros((3, 3)), Rbs.T, np.zeros((3, 2))), axis=1)  # Eq.(38)
    E2 = np.concatenate(
        (A3, A4, np.zeros((3, 3)), np.zeros((3, 3)), np.dot(w2wx(para.rcOc), Rbs.T), np.dot(Rbs.T, e_t)), axis=1)
    E3 = np.concatenate(
        (np.zeros((3, 3)), np.zeros((3, 3)), para.mp * np.eye(3), np.zeros((3, 3)), -np.eye(3), np.zeros((3, 2))),
        axis=1)
    E4 = np.concatenate((np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), Ip, -w2wx(para.rcOp), -e_t), axis=1)
    E5 = np.concatenate(
        (np.eye(3), -w2wx(para.rcOc), -Rbs.T, np.dot(Rbs.T, w2wx(para.rcOp)), np.zeros((3, 3)), np.zeros((3, 2))),
        axis=1)
    E6 = np.concatenate((np.zeros((1, 3)), -K82, np.zeros((1, 3)), K81, np.zeros((1, 3)), np.zeros((1, 2))), axis=1)
    E7_1 = np.dot(np.dot(np.array([[0, 0, 1]]), w2wx(para.rcOc)) + para.k_psi * psir, Rbs.T)
    E7_2 = np.dot(-np.dot(np.array([[0, 0, 1]]), Rbs.T), e_t)
    E7 = np.concatenate((np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3)), E7_1, E7_2), axis=1)

    B1_1 = np.dot(A1, v0) + np.dot(A2, w)  # Eq.(39)
    B1 = Faero + Fcg - np.dot(w2wx(w), B1_1)
    B2_1 = np.dot(np.dot(w2wx(v0), Ir[0:3, 3:6]), w) * 0
    B2_2_1 = np.dot(Ir[3:6, 0:3], v0)
    B2_2_2 = np.dot(Ir[3:6, 3:6], w)
    B2_2 = np.dot(w2wx(w), B2_2_1 + B2_2_2)
    B2 = Maero - B2_1 - B2_2 + mcf
    B3 = Fsa + Fsg - np.dot(w2wx(ws) * para.mp, vp) + para.thrust
    B4 = - np.dot(np.dot(w2wx(ws), Ip), ws) + mpf
    B5_1 = np.dot(Rbs.T, np.dot(w2wx(ws), (vp - np.dot(w2wx(para.rcOp), ws))))
    B5_2 = np.dot(w2wx(w), (v0 - np.dot(w2wx(para.rcOc), w)))
    B5 = B5_1 - B5_2
    B6 = dps  # ok
    B7 = np.array([Mcz])  # ok
    #调试
    # print('w2wx(ws) * para.mp',para.vw.shape)
    # print('B1',B1)
    # print('B1shape',B1.ndim)
    # print('B2',B2.ndim)
    # print('B3',B3.ndim)
    # print('B4',B4.ndim)
    # print('B5',B5.ndim)
    # print('B6',B6.ndim)
    # print('B7',dpsir.ndim)
    # print(dpsir)
    # print(tmp2)

    EE = np.concatenate((E1, E2, E3, E4, E5, E6, E7), axis=0)
    BB = np.concatenate((B1, B2, B3, B4, B5, B6, B7), axis=0)

    EE_ = EE.astype(np.float64)
    BB_ = BB.astype(np.float64)
    dVW = np.linalg.solve(EE_, BB_)  # More numerically stable than matrix inverse
    #print(dVW)
    # dVW = EE\BB
    # dVW包含伞体速度、伞体角速度、负载速度、负载角速度

    d_angle = np.dot(omega2angle_rate(angle[2][0], angle[1][0], angle[0][0]), w)
    # w是伞体角速度，angle是伞体姿态角
    # d_angle是下一个时刻得伞体姿态角
    vd = np.dot(np.dot(angle2dcm(0, 0, math.pi).T, Rnb.T), v0)
    # vd是翼伞系统位置;转换矩阵Ti-c，惯性（大地）坐标系到伞体坐标系;v0是伞体速度
    tmp = np.zeros((20, 1))
    tmp[0:3] = vd  # 伞体坐标系原点位置（翼伞系统位置）
    tmp[3:6] = d_angle  # 伞体姿态角
    tmp[6:8] = np.array([[dthetar, dpsir[0]]]).T  # 相对俯仰角和相对偏航角
    tmp[8:] = dVW[0:12]  # dVW包含伞体速度、伞体角速度、负载速度、负载角速度
    dydt = tmp
    dydt1=dydt.ravel()
    #print((dydt.ravel()).shape)
    return dydt1
# 注意，代码中thetar、psir、dthetar都是数字，而非数组

#


def canopy_inertial(para): #根据para参数求解伞体转动惯量
    Js11 = pow(para.r,2) - pow(para.sloc,2)
    rou1 = 3/(para.r*para.ca*para.c)
    rou2 = 1/(3*para.r*para.ca*para.c)
    Js13 = rou1*pow(para.r,2)*pow(para.c,2)*math.sin(para.ca/2)/16 - rou2*9*pow(para.r,2)*pow(para.c,2)*math.sin(para.ca/2)/16
    Js22= pow(para.r,2)*(math.sin(para.ca) + para.ca)/para.ca/2 + 7*pow(para.c,2)/48 - pow(para.sloc,2)
    Js31=Js13
    Js33=pow(para.r,2)*(para.ca - math.sin(para.ca))/2/para.ca + pow(para.c,2)*7/48
    m_air = para.Rho*para.b*para.c*para.t/2
    m_total = para.mc + m_air
    Ic=m_total*np.array([[Js11,0,Js13],[0,Js22,0],[Js31,0,Js33]])
    return Ic

def apparent_mass(para): #计算附加质量函数
    B2=np.array([[0,0,0],[0,1,0],[0,0,0]])
    jyuan = para.ca/2
    #youh = jyuan/4
    b = 2*para.b/para.ca*math.sin(para.ca/2)
    youh = para.r*(1 - math.cos(para.ca/2))/b
    AR = para.b/para.c
    Ka = 0.848
    Kb = 1.0
    Kc = AR/(1+AR)
    Ka_s = 0.84*Kc
    Kb_s = 1.64*Kc
    Kc_s = 0.848
    af11 = (para.Rho * Ka * math.pi * pow(para.t, 2) * b) / 4
    af22 = (para.Rho * Kb * math.pi * pow(para.t, 2) * para.c) / 4
    af33 = para.Rho * Kc * math.pi * pow(para.c, 2) * b / 4
    af44 = para.Rho * Ka_s * math.pi * pow(b, 3) * pow(para.c, 2) / 48
    af55 = para.Rho * Kb_s * pow(para.c, 4) * b / (12 * math.pi)
    af66 = para.Rho * Kc_s * pow(b, 3) * pow(para.t, 2) * math.pi / 48  # z轴的附加转动惯量
    zpc = para.r * math.sin(jyuan) / jyuan
    zrc = (zpc * af22) / (af22 + af44 / pow(para.r, 2))
    zpr = zrc - zpc
    ma11 = (1 + (8 / 3) * pow(youh, 2)) * af11
    ma22 = (pow(para.r,2) * af22 + af44) / pow(zpc, 2)
    ma33 = af33 * math.sqrt(1 + 2 * pow(youh, 2) * (1 - pow((para.t / para.c), 2)))
    xor = 0
    yor = 0
    zor = para.sloc - zrc
    xrp = 0
    yrp = 0
    zrp = zpr
    lor = np.array([[0, -zor, yor], [zor, 0, -xor], [-yor, xor, 0]])
    lrp = np.array([[0, -zrp, yrp], [zrp, 0, -xrp], [-yrp, xrp, 0]])
    ma = np.array([[ma11, 0, 0], [0, ma22, 0], [0, 0, ma33]])
    D = (lor + np.dot(lrp,B2))
    mal = np.dot(-ma,D)
    Tmal = np.dot((np.dot(B2,lrp) + lor) ,ma)
    Ia11 = (pow(zpr, 2) / pow(zpc, 2)) * pow(para.r, 2) * af22 + (pow(zrc, 2) / pow(zpc, 2)) * af44
    Ia22 = af55 * (1 + math.pi / 6 * (1 + AR) * AR * pow(youh,2) * pow((para.t / para.c), 2))
    Ia33 = (1 + 8 * pow(youh, 2)) * af66
    Ja = np.array([[Ia11, 0, 0], [0, Ia22, 0], [0, 0, Ia33]],dtype=object)
    Q = np.dot(np.dot(np.dot(B2,lrp),ma),lor)
    Jao = Ja - np.dot(np.dot(lor,ma) ,lor) - np.dot(np.dot(np.dot(lrp ,ma) ,lrp) , B2) - Q - Q.T
    Ic_a1 = np.concatenate((ma, mal),axis=1)
    Ic_a2 = np.concatenate((Tmal, Jao), axis=1)
    Ic_a = np.concatenate((Ic_a1, Ic_a2), axis=0)
    #Ic_a = np.array([ma, mal], [Tmal, Jao])
    return Ic_a
#print(apparent_mass(para))

def angle2dcm(phi, theta, psi):
    # 构建旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(phi), -np.sin(phi)],
                   [0, np.sin(phi), np.cos(phi)]])

    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])

    Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                   [np.sin(psi), np.cos(psi), 0],
                   [0, 0, 1]])
    DCM = np.dot(Rz, np.dot(Ry, Rx))
    DCM = np.array(
        [[DCM[2][2], DCM[2][1], DCM[2][0]], [DCM[1][2], DCM[1][1], DCM[1][0]], [DCM[0][2], DCM[0][1], DCM[0][0]]])
    return DCM

# R = angle2dcm(1, 0.2, 0.3)
# print(R)
