import os
os.system('cls' if os.name == 'nt' else 'clear')
import CoolProp.CoolProp as cp
# cesta k REFPROP (změň podle svého počítače)
os.environ["RPPREFIX"] = r"C:\Program Files\REFPROP"
import numpy as np
import matplotlib.pyplot as plt
import time
start = time.time()  # začátek měření

# vstupní parametry
T_evap = 273.15 + 80    # teplota v K
T_cond = 273.15 + 130   # teplota v K
dT_SH = 5               # přehřátí v K
dT_SC = 5               # podchlazení v K
eta_comp = 0.75         # účinnost kompresoru
Q_out = 500000          # požadovaný výkon v W

# definice směsi
latka1 = "R1233ZDE"
latka2 = "R1234ZEZ"

n = 25 # počet kroků pro výpočet
podil=np.linspace(0,1,1+n)

# inicializace polí pro výsledky
A = np.array([["Tlak [Pa]","Teplota [K]","Entalpie [J/kg]","Entropie [J/kg/K]","Hustota [kg/m3]"]])
B = np.array([["Stav","1","2","2s","3","4"]]).reshape(6,1)
F = np.array([["q_in [J/kg]","q_out [J/kg]","m_dot [kg/s]","w_cycle [J/kg]","W_comp [W]","COP [-]","VHC [J/m3]"]]).reshape(7,1)
J = np.array([["Látka","Podíl"]]).reshape(1,2)
ListE = []
ListH = []
ListL = []

for i in range(n+1):

    # výpočet molárního poměru pro směs
    podil2 = podil[i]
    podil1 = 1 - podil2
    směs = f"REFPROP::{latka1}[{podil1}]&{latka2}[{podil2}]"

    K = np.array([latka1,podil1,latka2,podil2]).reshape(2,2)
    L = np.vstack((J,K))

    # výpočty pro jednotlivé stavy
    T1 = T_evap + dT_SH
    T3 = T_cond - dT_SC
    P1 = cp.PropsSI("P","T", T_evap,"Q", 1,směs)
    h1 = cp.PropsSI("H","P", P1,"T", T1,směs)
    s1 = cp.PropsSI("S","P", P1,"T", T1,směs)
    ro1 = cp.PropsSI("D","P", P1,"H", h1,směs)
    s2s = s1
    P3 = cp.PropsSI("P","T", T_cond,"Q", 0,směs)
    P2s = P3
    h2s = cp.PropsSI("H","P", P2s,"S", s2s,směs)
    T2s = cp.PropsSI("T","P", P2s,"S", s2s,směs)
    ro2s = cp.PropsSI("D","P", P2s,"H", h2s,směs)
    h2 = h1 + (h2s - h1) / eta_comp
    P2 = P2s
    T2 = cp.PropsSI("T","P", P2,"H", h2,směs)
    s2 = cp.PropsSI("S","P", P2,"H", h2,směs)
    ro2 = cp.PropsSI("D","P", P2,"H", h2,směs)
    h3 = cp.PropsSI("H","P", P3,"T", T3,směs)
    s3 = cp.PropsSI("S","P", P3,"T", T3,směs)
    ro3 = cp.PropsSI("D","P", P3,"H", h3,směs)
    h4 = h3
    P4 = P1
    T4 = cp.PropsSI("T","P", P4,"H", h4,směs)
    s4 = cp.PropsSI("S","P", P4,"H", h4,směs)
    ro4 = cp.PropsSI("D","P", P4,"H", h4,směs)

    C = np.array([P1,T1,h1,s1,ro1,P2,T2,h2,s2,ro2,P2s,T2s,h2s,s2s,ro2s,P3,T3,h3,s3,ro3,P4,T4,h4,s4,ro4]).reshape(5,5)
    D = np.vstack((A,C))
    E = np.hstack((B,D))

    # výpočty směsi
    q_in = h1 - h4 # vstupní teplo
    q_out = h2 - h3 # výstupní teplo
    m_dot = Q_out / q_out # hmotnostní průtok
    w_cycle = q_out - q_in # práce cyklu
    W_comp = w_cycle * m_dot # výkon kompresoru
    COP = q_out / w_cycle # topný faktor
    VHC = q_out * ro1 # objemová topivost

    G = np.array([q_in, q_out, m_dot, w_cycle, W_comp, COP, VHC]).reshape(7,1)
    H = np.hstack((F,G))

    # zápis výsledků
    ListE.append(E)
    ListH.append(H)
    ListL.append(L)

Stav = np.stack(ListE, axis=2)
Obeh = np.stack(ListH, axis=2)
Smes = np.stack(ListL, axis=2)

end1 = time.time()  # průběžný konec měření
print(f"Výpočet trval {end1 - start} sekund")

# zobrazení výsledků pro COP a VHC
x = np.array(podil)
yCOP = np.array(Obeh[5,1,:]).astype(float)
yVHC = np.array(Obeh[6,1,:]).astype(float) / 1000 # převod z J/m3 na kJ/m3
# vytvoření grafu
fig, ax1 = plt.subplots()
# první křivka, osa vlevo
ax1.plot(x, yCOP, 'b-', label='COP [-]')
ax1.set_xlabel(f'Podíl složky {latka1}')
ax1.set_ylabel('COP [-]', color='b')
ax1.tick_params(axis='y', labelcolor='b')
# druhá křivka, osa vpravo
ax2 = ax1.twinx()  # sdílí osu x
ax2.plot(x, yVHC, 'r--', label='VHC [kJ/m3]')
ax2.set_ylabel('VHC [kJ/m3]', color='r')
ax2.tick_params(axis='y', labelcolor='r')
# přidání legendy
fig.tight_layout()  # aby se osy nepřekrývaly
plt.show()

end2 = time.time()  # konec měření
print(f"Graf trval {end2 - end1} sekund")

end = time.time()  # konec měření
print(f"Program trval {end - start} sekund")
