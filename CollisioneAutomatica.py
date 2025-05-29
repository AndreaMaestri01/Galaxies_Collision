import numpy as np
import astropy.units as unit
from astropy.constants import G, kpc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


#Funzione parametri crea un dizionario con i parametri della galassia
def parameters(*args):
    if len(args) != 8:
        print("Errore: sono necessari 8 argomenti")
    else:
        try:
            if len(args[2]) == 3 and len(args[3]) == 3 and len(args[4]) == 3:
                return { "mass"       : args[0]*unit.M_sun,
                         "radius"     : args[1]*unit.kpc,
                         "center_pos" : args[2]*unit.kpc,
                         "center_vel" : args[3]*unit.km/unit.s,
                         "normal"     : args[4],
                         "N_rings"    : args[5],
                         "N_stars"    : args[6],
                         "softening"  : args[7] }
            else:
                print("Errore: il 3., 4., e 5. argomento devono essere una tupla di 3 elementi")
        except TypeError:
            print("Errore: argomento non valido")

#Funzione che inizializza il disco galattico
def init_disk(galaxy, time_step=0.1*unit.Myr):

    dr = (1 - galaxy['softening'])*galaxy['radius']/galaxy['N_rings'] # lunghezza di un anello
    N_stars_per_ring = int(galaxy['N_stars']/galaxy['N_rings']) #numero di stelle per anello

    ###Rotazione di Rorigues###
    norm = np.sqrt(galaxy['normal'][0]**2 + galaxy['normal'][1]**2 + galaxy['normal'][2]**2) #fattore di normalizzazione dell'inclinazione
    cos_theta = galaxy['normal'][2]/norm
    sin_theta = np.sqrt(1-cos_theta**2)
    u = np.cross([0,0,1], galaxy['normal']/norm) #asse di rotazione 
    norm = np.sqrt(u[0]**2 + u[1]**2 + u[2]**2)

    if norm > 0:
        u /= norm #normalizzo

        # rotazione da piano galattico all'oservatore
        rotation = [[u[0]*u[0]*(1-cos_theta) + cos_theta,
                     u[0]*u[1]*(1-cos_theta) - u[2]*sin_theta,
                     u[0]*u[2]*(1-cos_theta) + u[1]*sin_theta],
                     [u[1]*u[0]*(1-cos_theta) + u[2]*sin_theta,
                      u[1]*u[1]*(1-cos_theta) + cos_theta,
                      u[1]*u[2]*(1-cos_theta) - u[0]*sin_theta],
                     [u[2]*u[0]*(1-cos_theta) - u[1]*sin_theta,
                      u[2]*u[1]*(1-cos_theta) + u[0]*sin_theta,
                      u[2]*u[2]*(1-cos_theta) + cos_theta]]

        #Printo angoli che definiscono l'orientamento della galassia
        phi = np.arctan2(galaxy['normal'][1], galaxy['normal'][0])
        theta = np.arccos(cos_theta)
        print("Plane normal: phi = {:.1f}°, theta = {:.1f}°".\
              format(np.degrees(phi), np.degrees(theta)))
        
        #caso in cui non serva rotazione 
    else:
        rotation = np.identity(3)

        #posizioni e velocità da tuple a np.array
    galaxy['stars_pos'] = np.array([])
    galaxy['stars_vel'] = np.array([])

    # Raggio interno dato da softening*raggio
    R = galaxy['softening']*galaxy['radius']

    for n in range(galaxy['N_rings']):

        # Posizione le stelle nei vari anelli in modo randomico
        r_star = R + dr * np.random.random_sample(size=N_stars_per_ring)
        phi_star = 2*np.pi * np.random.random_sample(size=N_stars_per_ring)

        # adopero per ciascuna stella la rotazione di Rodrigues
        vec_r = np.dot(rotation,
                       r_star*[np.cos(phi_star),
                               np.sin(phi_star),
                               np.zeros(N_stars_per_ring)])
        x = galaxy['center_pos'][0] + vec_r[0] #Trovo coordinate stelle nel sistema di riferimento del cm
        y = galaxy['center_pos'][1] + vec_r[1]
        z = galaxy['center_pos'][2] + vec_r[2]

        #Terza legge di Keplero e Leapfrog scheme
       
        T_star = 2*np.pi * ((G*galaxy['mass'])**(-1/2) * r_star**(3/2)).to(unit.s) #legge di Keplero 
        delta_phi = 2*np.pi * time_step.to(unit.s).value / T_star.value

        vec_v = np.dot(rotation,
                       (r_star.to(unit.km)/time_step.to(unit.s)) *
                       [(np.cos(phi_star) - np.cos(phi_star - delta_phi)),
                        (np.sin(phi_star) - np.sin(phi_star - delta_phi)),
                        np.zeros(N_stars_per_ring)])
        v_x = galaxy['center_vel'][0] + vec_v[0]
        v_y = galaxy['center_vel'][1] + vec_v[1]
        v_z = galaxy['center_vel'][2] + vec_v[2]

    #estendo il dizionario galassia(salvato fuori da funzione):

        if galaxy['stars_pos'].size == 0:
            galaxy['stars_pos'] = np.array([x,y,z])
            galaxy['stars_vel'] = np.array([v_x,v_y,v_z])
        else:
            galaxy['stars_pos'] = np.append(galaxy['stars_pos'], np.array([x,y,z]), axis=1)
            galaxy['stars_vel'] = np.append(galaxy['stars_vel'], np.array([v_x,v_y,v_z]), axis=1)

        R += dr

    # Ripristino unità misura
    galaxy['stars_pos'] *= unit.kpc
    galaxy['stars_vel'] *= unit.km/unit.s

    # introduco velocità caratteristica definita dalla velocità di Kepler a metà del raggio del disco
    galaxy['vel_scale'] = np.sqrt(G*galaxy['mass']/(0.5*R)).to(unit.km/unit.s)

#Funzione che fa evolvere i due dischi
def evolve_two_disks(primary, secondary, N_steps=1000, N_snapshots=100,time_step=0.1*unit.Myr):
   
    dt = time_step.to(unit.s).value #inizializzo step evoluzione temporale

    #introduco valori minimi di distanza per softening
    r_min1 = primary['softening']*primary['radius'].to(unit.m).value 
    r_min2 = secondary['softening']*secondary['radius'].to(unit.m).value

    N1, N2 = primary['N_stars'], secondary['N_stars']

    # massa, posizione and velocita prima galassia
    M1 = primary['mass'].to(unit.kg).value
    X1, Y1, Z1 = primary['center_pos'].to(unit.m).value
    V1_x, V1_y, V1_z = primary['center_vel'].to(unit.m/unit.s).value

    # massa, posizione and velocita prima galassia
    M2 = secondary['mass'].to(unit.kg).value
    X2, Y2, Z2 = secondary['center_pos'].to(unit.m).value
    V2_x, V2_y, V2_z = secondary['center_vel'].to(unit.m/unit.s).value

    # posizioni stelle prima galassia
    x = primary['stars_pos'][0].to(unit.m).value
    y = primary['stars_pos'][1].to(unit.m).value
    z = primary['stars_pos'][2].to(unit.m).value

    # posizioni stelle seconda galassia
    x = np.append(x, secondary['stars_pos'][0].to(unit.m).value)
    y = np.append(y, secondary['stars_pos'][1].to(unit.m).value)
    z = np.append(z, secondary['stars_pos'][2].to(unit.m).value)

    # velocità stelle prima galassia
    v_x = primary['stars_vel'][0].to(unit.m/unit.s).value
    v_y = primary['stars_vel'][1].to(unit.m/unit.s).value
    v_z = primary['stars_vel'][2].to(unit.m/unit.s).value

    # velocità stelle seconda galassia 
    v_x = np.append(v_x, secondary['stars_vel'][0].to(unit.m/unit.s).value)
    v_y = np.append(v_y, secondary['stars_vel'][1].to(unit.m/unit.s).value)
    v_z = np.append(v_z, secondary['stars_vel'][2].to(unit.m/unit.s).value)

    # introduco array per memorizzare evoluzione
    snapshots = np.zeros(shape=(N_snapshots+1,3,N1+N2+2))
    snapshots[0] = [np.append([X1,X2], x), np.append([Y1,Y2], y), np.append([Z1,Z2], z)]
    #print(snapshots.shape)

    # definisco il numero di steps per snapshot
    div = max(int(N_steps/N_snapshots), 1)

    print("Solving equations of motion for two galaxies (Leapfrog integration)")

    for n in range(1,N_steps+1):

        # distanza radiale dal cm con softening
        r1 = np.maximum(np.sqrt((X1 - x)**2 + (Y1 - y)**2 + (Z1 - z)**2), r_min1)
        r2 = np.maximum(np.sqrt((X2 - x)**2 + (Y2 - y)**2 + (Z2 - z)**2), r_min2)
        #print("\nr {:.6e} {:.6e} {:.6e} {:.6e}".format(r1[0],r2[0],r1[N1],r2[N1]))

        # aggiorno velocità stelle (dovuta alla gravità cm)
        v_x += G.value*(M1*(X1 - x)/r1**3 + M2*(X2 - x)/r2**3) * dt
        v_y += G.value*(M1*(Y1 - y)/r1**3 + M2*(Y2 - y)/r2**3) * dt
        v_z += G.value*(M1*(Z1 - z)/r1**3 + M2*(Z2 - z)/r2**3) * dt
        

        # aggiorno posizione stelle
        x += v_x*dt
        y += v_y*dt
        z += v_z*dt
        
        ##Simulo dinamica due corpi##
        # distanza tra i due centri
        D_sqr_min = (r_min1+r_min2)**2
        D_cubed = (max((X1 - X2)**2 + (Y1 - Y2)**2 + (Z1 - Z2)**2, D_sqr_min))**(3/2)

        # Accelerazione Gravitazionale prima galassia
        A1_x = G.value*M2*(X2 - X1)/D_cubed
        A1_y = G.value*M2*(Y2 - Y1)/D_cubed
        A1_z = G.value*M2*(Z2 - Z1)/D_cubed

        # Aggiorno velocità dei centri (velocità cm costante)
        V1_x += A1_x*dt; V2_x -= (M1/M2)*A1_x*dt
        V1_y += A1_y*dt; V2_y -= (M1/M2)*A1_y*dt
        V1_z += A1_z*dt; V2_z -= (M1/M2)*A1_z*dt
    

        # Aggiorno posizione dei centri
        X1 += V1_x*dt; X2 += V2_x*dt
        Y1 += V1_y*dt; Y2 += V2_y*dt
        Z1 += V1_z*dt; Z2 += V2_z*dt
        
        ##catturo snapshot#

        if n % div == 0:
            i = int(n/div)
            snapshots[i] = [np.append([X1,X2], x), np.append([Y1,Y2], y), np.append([Z1,Z2], z)]

        # caricamento
        print("\r{:3d} %".format(int(100*n/N_steps)), end="")
    #array tempi snapshot
    time = np.linspace(0*time_step, N_steps*time_step, N_snapshots+1, endpoint=True)
    print(" (stopped at t = {:.1f})".format(time[-1]))

    #sistemo dimensione snnapshot
    snapshots *= unit.m
    return time, snapshots.to(unit.kpc)

#Funzione che plotta l'evoluzione
def show_two_disks_3d(snapshot, N1, xlim=None, ylim=None, zlim=None, time=None, name=None):

    N2 = snapshot.shape[1]-2-N1 #calcolo stelle secondo disco

    fig = plt.figure(figsize=(10,10), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    
    #assi 3D
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel(r'$x$ [kpc]', fontsize=12)
    ax.set_ylabel(r'$y$ [kpc]', fontsize=12)
    ax.set_zlabel(r'$z$ [kpc]', fontsize=12)

    #se forniti imposto limiti assi
    if xlim != None: ax.set_xlim(xlim[0], xlim[1])
    if ylim != None: ax.set_ylim(ylim[0], ylim[1])
    if zlim != None: ax.set_zlim(zlim[0], zlim[1])

    #se fornito imposto nome tempo
    if time != None:
        title = ax.set_title('$t$ = {:.1f}'.format(time))

    #plotto i centri galattici
    ax.scatter(snapshot[0,0:2],          snapshot[1,0:2],          snapshot[2,0:2], \
               marker='+', color='black') 
    #plotto stelle primo disco
    ax.scatter(snapshot[0,2:N1+2],       snapshot[1,2:N1+2],       snapshot[2,2:N1+2], \
               marker='.', color='blue', s=2)
    #plotto stelle secondo disco
    ax.scatter(snapshot[0,N1+2:N1+N2+2], snapshot[1,N1+2:N1+N2+2], snapshot[2,N1+2:N1+N2+2], \
               marker='.', color='red', s=2)
    #se fornito un nome salvo nel file
    if name != None:
        plt.savefig(name + '_{:.0f}.png'.format(time))

if __name__== '__main__':
   
    #Creo dizionario con le due galassie:
    galassie = {

    'galassia1' :parameters(
    # Massa gallassia 1 (Masse Solari):
    float(input("Inserisci la MASSA della Galassia 1 (Masse Solari): ")),
    # Raggio galassia 1(kpc):
    float(input("Inserisci il RAGGIO della Galassia 1 (kpc): ")),
    # Coordinate formato '(x,y,z)' posizione iniziale centro galassia 1 (kpc):
    tuple(map(float, input("Inserisci la POSIZIONE iniziale (X, Y, Z) in kpc della Galassia 1, separati da virgola: ").split(','))),
    #Componenti formato '(x,y,z)' velocità iniziale centro galassia 1 (km/s):
    tuple(map(float, input("Inserisci la VELOCITA iniziale (Vx, Vy, Vz) in km/s della Galassia 1, separati da virgola: ").split(','))),
    #Componenti formato '(x,y,z)' normale al piano galassia 1:
    tuple(map(float, input("Inserisci l'INCLINAZIONE iniziale (nx, ny, nz) della Galassia 1, separati da virgola: ").split(','))),
    #Numero di anelli:
    int(input("Inserisci numero di ANELLI della Galassia 1: ")),
    #Numero di stelle:
    int(input("Inserisci numero di STELLE della Galassia 1: ")),
    #Fattore di softening per inner edge:
    float(input("Inserisci il fattore di SOFTENING (0,1) della Galassia 1: "))),

    'galassia2' :parameters(
    # Massa gallassia 2 (Masse Solari):
    float(input("Inserisci la MASSA della Galassia 2 (Masse Solari): ")),
    # Raggio galassia 2(kpc):
    float(input("Inserisci il RAGGIO della Galassia 2 (kpc): ")),
    # Coordinate formato '(x,y,z)' posizione iniziale centro galassia 2 (kpc):
    tuple(map(float, input("Inserisci la POSIZIONE iniziale (X, Y, Z) in kpc della Galassia 2, separati da virgola: ").split(','))),
    #Componenti formato '(x,y,z)' velocità iniziale centro galassia 2 (km/s):
    tuple(map(float, input("Inserisci la VELOCITA iniziale (Vx, Vy, Vz) in km/s della Galassia 2, separati da virgola: ").split(','))),
    #Componenti formato '(x,y,z)' normale al piano galassia 2:
    tuple(map(float, input("Inserisci l'INCLINAZIONE iniziale (nx, ny, nz) della Galassia 2, separati da virgola: ").split(','))),
    #Numero di anelli:
    int(input("Inserisci numero di ANELLI della Galassia 2: ")),
    #Numero di stelle:
    int(input("Inserisci numero di STELLE della Galassia 2: ")),
    #Fattore di softening per inner edge:
    float(input("Inserisci il fattore di SOFTENING (0,1) della Galassia 2: ")))
    }

    #Inizializzo i due dischi galattici
    init_disk(galassie['galassia1'])
    init_disk(galassie['galassia2'])
    
    time_step=float(input("Inserisci lo Step Temporale (Myr): "))
    N_steps=int(input("Inserisci numero di Step Numerici: "))
    print(f"Evoluzione osservata per {time_step * N_steps} Myr")
    N_snapshots=int(input("Inserisci numero di Snapshot: "))
    
    
    #immagini prodotte
    N_image=int(input("Inserisci numero di Immagini prodotte: "))
    
    #Evolvo i due dischi
    t, data = evolve_two_disks(galassie['galassia1'], galassie['galassia2'], N_steps=N_steps, N_snapshots=N_snapshots, time_step=time_step*unit.Myr)
    
    max_x = np.max(np.abs(data[:, 0, :]), axis=(0, 1)).value
    max_y = np.max(np.abs(data[:, 1, :]), axis=(0, 1)).value
    max_z = np.max(np.abs(data[:, 2, :]), axis=(0, 1)).value
    
    for i in range(0,N_snapshots,int(N_snapshots/N_image)):
        show_two_disks_3d(data[i,:,:], galassie['galassia1']['N_stars'],[-max_x,max_x], [-max_y,max_y], [-max_z,max_z], t[i],'Collisione')