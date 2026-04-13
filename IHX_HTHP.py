import os
os.system('cls' if os.name == 'nt' else 'clear')
import CoolProp.CoolProp as cp
# cesta k REFPROP (změň podle svého počítače)
os.environ["RPPREFIX"] = r"C:\Program Files\REFPROP"
import numpy as np
import matplotlib.pyplot as plt
import time
from tabulate import tabulate as tabulate
start = time.time()  # začátek měření

# vstupní parametry
T_evap = 273.15 + 80    # teplota v K
T_cond = 273.15 + 130   # teplota v K
dT_SH = 5               # přehřátí v K
dT_SC = 5               # podchlazení v K
eta_comp = 0.75         # účinnost kompresoru
Q_out = 500000          # požadovaný výkon v W
dT_IHX = 20             # teplotní rozdíl ve vnitřním výměníku v K

# definice směsi
latka1 = "R1233ZDE"
latka2 = "R1234ZEZ"

n = 100 # počet kroků pro výpočet
podil=np.linspace(0,1,1+n)

# inicializace polí pro výsledky
#A = np.array([["Tlak [Pa]","Teplota [K]","Entalpie [J/kg]","Entropie [J/kg/K]","Hustota [kg/m3]"]])
#B = np.array([["Stav","1","2","2s","3","4"]]).reshape(6,1)
#F = np.array([["q_in [J/kg]","q_out [J/kg]","m_dot [kg/s]","w_cycle [J/kg]","W_comp [W]","COP [-]","VHC [J/m3]"]]).reshape(7,1)
#J = np.array([["Látka","Podíl"]]).reshape(1,2)
ListStav = []
ListObeh = []
ListSmes = []

for i in range(n+1):

    # výpočet molárního poměru pro směs
    podil2 = podil[i]
    podil1 = 1 - podil2
    směs = f"REFPROP::{latka1}[{podil1}]&{latka2}[{podil2}]"
    Sm = np.array([latka1,podil1,latka2,podil2]).reshape(2,2)

    # výpočty pro jednotlivé stavy
    T1 = T_evap + dT_SH
    T3 = T_cond - dT_SC
    P1 = cp.PropsSI("P","T", T_evap,"Q", 1,směs)
    h1 = cp.PropsSI("H","P", P1,"T", T1,směs)
    s1 = cp.PropsSI("S","P", P1,"T", T1,směs)
    ro1 = cp.PropsSI("D","P", P1,"H", h1,směs)
    P1ihx = P1
    T1ihx = T3 - dT_IHX
    h1ihx = cp.PropsSI("H","P", P1ihx,"T", T1ihx,směs)
    s1ihx = cp.PropsSI("S","P", P1ihx,"T", T1ihx,směs)
    ro1ihx = cp.PropsSI("D","P", P1ihx,"H", h1ihx,směs)
    s2s = s1ihx
    P3 = cp.PropsSI("P","T", T_cond,"Q", 0,směs)
    P2s = P3
    h2s = cp.PropsSI("H","P", P2s,"S", s2s,směs)
    T2s = cp.PropsSI("T","P", P2s,"S", s2s,směs)
    ro2s = cp.PropsSI("D","P", P2s,"H", h2s,směs)
    h2 = h1ihx + (h2s - h1ihx) / eta_comp
    P2 = P2s
    T2 = cp.PropsSI("T","P", P2,"H", h2,směs)
    s2 = cp.PropsSI("S","P", P2,"H", h2,směs)
    ro2 = cp.PropsSI("D","P", P2,"H", h2,směs)
    h3 = cp.PropsSI("H","P", P3,"T", T3,směs)
    s3 = cp.PropsSI("S","P", P3,"T", T3,směs)
    ro3 = cp.PropsSI("D","P", P3,"H", h3,směs)
    P3ihx = P3
    h3ihx = h3 - (h1ihx - h1) # předpoklad: bez tepelných ztrát ve vnitřním výměníku
    T3ihx = cp.PropsSI("T","P", P3ihx,"H", h3ihx,směs)
    s3ihx = cp.PropsSI("S","P", P3ihx,"T", T3ihx,směs)
    ro3ihx = cp.PropsSI("D","P", P3ihx,"H", h3ihx,směs)
    h4 = h3ihx
    P4 = P1
    T4 = cp.PropsSI("T","P", P4,"H", h4,směs)
    s4 = cp.PropsSI("S","P", P4,"H", h4,směs)
    ro4 = cp.PropsSI("D","P", P4,"H", h4,směs)
    St = np.array([P1,T1,h1,s1,ro1,P1ihx,T1ihx,h1ihx,s1ihx,ro1ihx,P2,T2,h2,s2,ro2,P2s,T2s,h2s,s2s,ro2s,P3,T3,h3,s3,ro3,P3ihx,T3ihx,h3ihx,s3ihx,ro3ihx,P4,T4,h4,s4,ro4]).reshape(7,5)

    # výpočty směsi
    q_in = h1 - h4 # vstupní teplo
    q_out = h2 - h3 # výstupní teplo
    m_dot = Q_out / q_out # hmotnostní průtok
    w_cycle = q_out - q_in # práce cyklu
    W_comp = w_cycle * m_dot # výkon kompresoru
    COP = q_out / w_cycle # topný faktor
    VHC = q_out * ro1ihx # objemová topivost
    Ob = np.array([q_in, q_out, m_dot, w_cycle, W_comp, COP, VHC]).reshape(7,1)

    # zápis výsledků
    ListStav.append(St)
    ListObeh.append(Ob)
    ListSmes.append(Sm)

Stav = np.stack(ListStav, axis=2)
Obeh = np.stack(ListObeh, axis=2)
Smes = np.stack(ListSmes, axis=2)

end1 = time.time()  # průběžný konec měření
print(f"Výpočet trval {end1 - start} sekund")

# zobrazení výsledků pro COP a VHC
x = np.array(podil)
yCOP = np.array(Obeh[5,0,:]).astype(float)
yVHC = np.array(Obeh[6,0,:]).astype(float) / 1000 # převod z J/m3 na kJ/m3
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

#arr = np.array([[10, 20, 30],[40, 50, 60]])
#print(tabulate(arr,headers=["Sloupec A", "Sloupec B", "Sloupec C"],showindex=["První", "Druhý"],tablefmt="grid"))