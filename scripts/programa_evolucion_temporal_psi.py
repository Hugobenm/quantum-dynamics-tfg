import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ESTÉTICA DEFINITIVA ---
# 1. Configuramos para usar LaTeX real
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"], # O "Computer Modern Roman"
    "font.size": 11,                   # 10 u 11 es estándar para papers
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 12,
})

# Definimos el espacio
Lx = 5
Ly = 5
N = 128
x = np.linspace(-Lx, Lx, N, endpoint=False)
y = np.linspace(-Ly, Ly, N, endpoint=False)
X, Y = np.meshgrid(x, y)
dx = x[1] - x[0]
dy = y[1] - y[0]

# Por si queremos definir una función de onda nosotros mismos
sigma = 2.5
psi = np.zeros((2, N, N), dtype=complex)
# Acceder a la componente UP:
psi[0, :, :] = np.exp(-(X**2)/(2*sigma**2)) # O simplemente psi[0]
# Acceder a la componente DOWN:
psi[1, :, :] = np.zeros_like(X) # O simplemente psi[1]
#Normalizamos
def normalizar(p):
    norm = np.sqrt(np.sum(np.abs(p)**2)*dx*dy)
    return p / norm
psi = normalizar(psi)

# Si en lugar queremos cargar una función de onda de un fichero
psi = np.load('datos_tfg_gaugeLandau_g=0.npz')['psi']

g  = 0 # Fuerza de la interacción gp
def calcular_potencial_gp(psi, g):
    densidad = np.sum(np.abs(psi)**2, axis = 0)
    return g * densidad

#Definimos el grid en el espacio de momentos
def get_k_grid(N, Lx, Ly):
    """
    Genera la red en el espacio de momentos (k).
    Numpy ordena las frecuencias de una forma particular:
    [0, 1, ..., N/2 - 1, -N, ..., -1]
    """
    # Frecuencias kx y ky
    kx = np.fft.fftfreq(N, d=2*Lx/N) * 2 * np.pi
    ky = np.fft.fftfreq(N, d = 2*Ly/N) * 2 * np.pi

    # Creamos el meshgrid de momentos (igual que con X e Y)
    KX, KY = np.meshgrid(kx, ky)
    return KX, KY

# Ejecutamos la función para tener ya nuestro espacio de momentos
KX, KY = get_k_grid(N, Lx, Ly)

# Definimos el gauge
B = 1
E = 0.5
A_x = np.zeros_like(psi)
A_x[0, :, :] = 0
A_x[1, :, :] = 0
A_y = np.zeros((2, N, N), dtype=complex)
A_y[0, :, :] = B * X 
A_y[1, :, :] = B * X 
A_t = np.zeros((2, N, N), dtype=complex)
A_t[0, :, :] = E*X 
A_t[1, :, :] = E*X 

def Hamiltoniano_cinético(psi, A_x, A_y, A_t, KX, KY, dx, dy, hbar = 1, m=1, q=1):
    """
    Aplica H_cin = (-hbar^2/2m)Laplaciano + (ihbar q/m)A.Grad + (q^2/2m)A^2
    pero ahora usando derivadas espectrales (FFT). Más preciso y estable que np.roll (diferencias finitas)
    """
    #1. Pasamos al espacio de momentos 
    # axes = (1, 2) le dice que transforme las dimensiones espaciales x,y (ignorando el indice del spin, i.e. axis = 0)
    psi_k = np.fft.fft2(psi, axes=(1, 2))

    # 2. Calculamos las derivadas en el espacio k
    # Derivada primera en x es simplemente multiplicar por i*k_x (viene de la definición de momento en el espacio real)
    d_psi_x_k = (1j * KX) * psi_k
    d_psi_y_k = (1j * KY) * psi_k

    # Laplaciano (Derivada segunda: d2/dx2 -> *(-k_x^2)

    laplaciano_k = - (KX**2 + KY**2) * psi_k

    #3. Volvemos al espacio real (IFFT)
    d_psi_x = np.fft.ifft2(d_psi_x_k, axes = (1, 2))
    d_psi_y = np.fft.ifft2(d_psi_y_k, axes = (1, 2))
    laplaciano = np.fft.ifft2(laplaciano_k, axes = (1, 2))
    
    # Construimos el Hamiltoniano
    # Término A . Grad
    A_dot_grad = A_x * d_psi_x + A_y * d_psi_y

    # 3. Término cuadrático A^2
    A_sq = A_x**2 + A_y**2

    # JUNTAMOS TODO (Ojo a los signos y la i imaginaria)
    # H_cin_psi = - (h^2/2m)lap + i (hq/m) (A grad)+ (q^2 / 2m) A^2 psi

    term1 = - (hbar**2 / (2*m)) * laplaciano
    term2 = 1j * (hbar * q / m) * A_dot_grad  # El término magnético es complejo puro
    term3 = (q**2 / (2*m)) * A_sq * psi
    term4 = q*A_t*psi

    return term1 + term2 + term3 + term4

def H_spin(psi, B, hbar = 1, m = 1, q = 1):
    omega_B = q*B/m
    E_spin = np.zeros_like(psi)
    E_spin[0, :, :] = -hbar*omega_B/2*psi[0, :, :]
    E_spin[1, :, :] = hbar*omega_B/2*psi[1, :, :]
    return E_spin

# Añadimos trampa armónica para evitar conflictos en los bordes con las condiciones periódicas de contorno (asumidas por FFT): 
# EN ESTE CASO NO LA ESTAMOS USANDO PASAMOS A POZO DE POTENCIAL EN X, AÚN ASÍ LA DEJO POR AQUÍ POR SI ACASO
q = 1
m = 1
omega_B = q*B/m
omega_trap = 0.5*omega_B
def Trampa_armonica(omega_trap, m = 1):
    V_trap = 0.5 * m * (omega_trap**2) * (X**2 + Y **2)
    return V_trap
    
# Definimos el potencial de pozo que nos confinará la dirección X
V_0 = 9 * 1 * omega_B   # hbar = 1

def Pozo_potencial(V_0, Lx):
    X_max = Lx / 1.5        
    potencial_pozo = V_0 * (1 + np.tanh((np.abs(X) - 0.75*X_max) * 9 ))
    return potencial_pozo

# Evaluamos el potencial para no estar calculando en cada iteración
Potencial_pozo = Pozo_potencial(V_0, Lx)

def calcular_derivada_tiempo(psi, hbar = 1):
    # Caculamos el potencial gp
    H_gp = calcular_potencial_gp(psi, g) * psi

    #Calculamos el resto de términos
    H_kin = Hamiltoniano_cinético(psi, A_x, A_y, A_t, KX, KY, dx, dy)
    H_sp = H_spin(psi, B)
    H_poz = Potencial_pozo * psi

    H_total = H_gp + H_kin + H_sp + H_poz
    # Derivada total
    return -1j * H_total / hbar

def paso_RK4(psi, dt): # Solver Runge-Kutta 4
    k1 = calcular_derivada_tiempo(psi)
    k2 = calcular_derivada_tiempo(psi + 0.5 * dt * k1)
    k3 = calcular_derivada_tiempo(psi + 0.5 * dt * k2)
    k4 = calcular_derivada_tiempo(psi + dt * k3)

    return psi + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

# HACEMOS EL BUCLE DE LA EVOLUCIÓN TEMPORAL

dt = 0.001  # Intervalo de tiempo entre dos puntos
max_steps = 5000 # Cuanto tiempo queremos simular
fotogramas = []    # Aquí guardaremos la densidad para el vídeo

def densidad(psi):
    densidad_psi = np.abs(psi[0])**2 + np.abs(psi[1])**2
    return densidad_psi

# Partimos del estado que queramos, podemos cargar uno en concreto
fotogramas.append(densidad(psi))

for step in range(max_steps):
    psi = paso_RK4(psi, dt)

    # Preservamos la norma, porque si no diverge poco a poco
    psi = normalizar(psi)
    
    if step % 50 == 0:
        fotogramas.append(densidad(psi))

#### REPRESENTACIÓN GRÁFICA DE LA EVOLUCIÓN TEMPORAL

import matplotlib.animation as animation

fig_anim, ax_anim = plt.subplots(figsize=(5, 4))
ax_anim.set_xlabel(r'$x / l_B$')
ax_anim.set_ylabel(r'$y / l_B$')
ax_anim.set_title(r'Evolución temporal $|\psi|^2$')

# Pintamos el primer fotograma
cmap = plt.cm.inferno
im_anim = ax_anim.imshow(fotogramas[0], cmap=cmap, origin='lower', extent=[-Lx, Lx, -Ly, Ly], interpolation='bilinear', vmin=0, vmax=np.max(fotogramas[0]))
fig_anim.colorbar(im_anim, ax=ax_anim, label=r'$|\psi|^2$')

# Función que actualiza la imagen en cada frame del vídeo
def actualizar(frame):
    im_anim.set_data(fotogramas[frame])
    return [im_anim]


# Creamos la animación
ani = animation.FuncAnimation(fig_anim, actualizar, frames=len(fotogramas), blit=True)

#Guardamos como GIF (Hace falta tener pillow instalado)
print("Generando animación...")
ani.save('evolucion_tiempo_real.gif', writer='pillow', fps=15)
print("Vídeo guardado como evolucion_tiempo_real.gif")

plt.show()

# Tomamos el índice central de Y
idx_y = N // 2

plt.figure(figsize=(8, 5))

# 1. Pintamos el estado inicial (la montaña original de g=0)
plt.plot(x, fotogramas[0][idx_y, :], label='Inicio (t=0, g=0)', color='black', lw=2, linestyle='--')

# 2. Pintamos el estado a mitad de la simulación
plt.plot(x, fotogramas[len(fotogramas)//2][idx_y, :], label='Mitad de evolución', color='blue', lw=2)

# 3. Pintamos el estado final (la montaña de g=10)
plt.plot(x, fotogramas[-1][idx_y, :], label='Final (g=10)', color='red', lw=2)

plt.title("Evolución de la Densidad 1D (Corte Transversal)")
plt.xlabel(r'$x/l_B$')
plt.ylabel(r'Densidad $|\psi|^2$')
plt.xlim(-Lx/1.5, Lx/1.5)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()