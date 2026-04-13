import os
os.system('cls' if os.name == 'nt' else 'clear')
from multiprocessing import Pool, cpu_count
import CoolProp.CoolProp as cp
from CoolProp.CoolProp import AbstractState
# cesta k REFPROP (změň podle svého počítače)
os.environ["RPPREFIX"] = r"C:\Program Files\REFPROP"
import numpy as np
import time
from itertools import combinations
cesta = f"{os.getcwd()}\směsi\kvaternární směsi\data směsí"
os.makedirs(cesta, exist_ok=True)


start = time.time()  # začátek měření
# vstupní parametry
T_evap = 273.15 + 80    # teplota v K
T_cond = 273.15 + 130   # teplota v K
dT_SH = 5               # přehřátí v K
dT_SC = 5               # podchlazení v K
eta_comp = 0.75         # účinnost kompresoru
Q_out = 500000          # požadovaný výkon ve W

T1 = T_evap + dT_SH
T3 = T_cond - dT_SC

n = 20 # počet kroků pro výpočet
podil = np.linspace(0,1,n+1)

# inicializace polí pro výsledky
#A = np.array([["Tlak [Pa]","Teplota [K]","Entalpie [J/kg]","Entropie [J/kg/K]","Hustota [kg/m3]"]])
#B = np.array([["Stav","1","2","2s","3","4"]]).reshape(6,1)
#F = np.array([["q_in [J/kg]","q_out [J/kg]","m_dot [kg/s]","w_cycle [J/kg]","W_comp [W]","COP [-]","VHC [J/m3]"]]).reshape(7,1)
#J = np.array([["Látka","Podíl"]]).reshape(1,2)
Stav = np.empty((5, 5, n+1, n+1, n+1))
Obeh = np.empty((7, 1, n+1, n+1, n+1))
Podil = np.empty((4, 1, n+1, n+1, n+1))
poc = np.zeros((n+1))

with open(f"{os.getcwd()}\směsi\pure_fluids6.txt", "r") as f:
    latky = [line.strip() for line in f if line.strip()]
for komb in combinations(latky, 4):
    latka1, latka2, latka3, latka4 = komb
    Smes = np.array([[[latka1, latka2, latka3, latka4]]]).reshape(4,1)
    AS = AbstractState("REFPROP", f"{latka1}&{latka2}&{latka3}&{latka4}")

    for i in range(n+1):
        for j in range(n+1):
            for k in range(n+1):
                
                # výpočet molárního poměru pro směs
                podil4 = round(podil[i], 2)
                podil3 = round(podil[j], 2)
                podil2 = round(podil[k], 2)
                if podil2 + podil3 + podil4 > 1:
                    St = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).reshape(5,5)
                    Ob = np.array([0,0,0,0,0,0,0]).reshape(7,1)
                    Po = np.array([0,0,0,0]).reshape(4,1)
                    Stav[:,:,i,j,k] = St
                    Obeh[:,:,i,j,k] = Ob
                    Podil[:,:,i,j,k] = Po
                    poc[i] += 1
                    continue
                
                podil1 = round(1 - podil2 - podil3 - podil4, 2)
                Po = np.array([podil1,podil2,podil3,podil4]).reshape(4,1)
                AS.set_mole_fractions([podil1, podil2, podil3, podil4])
                
                # výpočty pro jednotlivé stavy
                #stav 1
                AS.update(cp.QT_INPUTS, 1, T_evap)
                P1 = AS.p()
                AS.update(cp.PT_INPUTS, P1, T1)
                h1 = AS.hmass()
                s1 = AS.smass()
                ro1 = AS.rhomass()
                #stav 3
                AS.update(cp.QT_INPUTS, 0, T_cond)
                P3 = AS.p()
                AS.update(cp.PT_INPUTS, P3, T3)
                h3 = AS.hmass()
                s3 = AS.smass()
                ro3 = AS.rhomass()
                #stav 2s
                s2s = s1
                P2s = P3
                AS.update(cp.PSmass_INPUTS, P2s, s2s)
                h2s = AS.hmass()
                T2s = AS.T()
                ro2s = AS.rhomass()
                #stav 2
                h2 = h1 + (h2s - h1) / eta_comp
                P2 = P2s
                AS.update(cp.HmassP_INPUTS, h2, P2)
                T2 = AS.T()
                s2 = AS.smass()
                ro2 = AS.rhomass()
                #stav 4
                h4 = h3
                P4 = P1
                try:
                    AS.update(cp.HmassP_INPUTS, h4, P4)
                except Exception:
                    print(f"Chyba při směsi = {latka1}&{latka2}&{latka3}&{latka4} a podílu {podil1},{podil2},{podil3},{podil4}")
                    pod= podil1+podil2+podil3+podil4
                    print(f"Součet podílů: {pod}")
                    continue
                T4 = AS.T()
                s4 = AS.smass()
                ro4 = AS.rhomass()
                St = np.array([P1,T1,h1,s1,ro1,P2,T2,h2,s2,ro2,P2s,T2s,h2s,s2s,ro2s,P3,T3,h3,s3,ro3,P4,T4,h4,s4,ro4]).reshape(5,5)
                
                # výpočty směsi
                q_in = h1 - h4 # vstupní teplo
                q_out = h2 - h3 # výstupní teplo
                m_dot = Q_out / q_out # hmotnostní průtok
                w_cycle = q_out - q_in # práce cyklu
                W_comp = w_cycle * m_dot # výkon kompresoru
                COP = q_out / w_cycle # topný faktor
                VHC = q_out * ro1 # objemová topivost
                Ob = np.array([q_in, q_out, m_dot, w_cycle, W_comp, COP, VHC]).reshape(7,1)

                # zápis výsledků
                Stav[:,:,i,j,k] = St
                Obeh[:,:,i,j,k] = Ob
                Podil[:,:,i,j,k] = Po
    del AS
    np.savez(f"{cesta}\{latka1}&{latka2}&{latka3}&{latka4}.npz", Smes = Smes, Podil = Podil, Stav = Stav, Obeh = Obeh)

print(f"Počet kombinací: {poc}")
end = time.time()  # konec měření
print(f"Výpočet trval {end - start} sekund")
