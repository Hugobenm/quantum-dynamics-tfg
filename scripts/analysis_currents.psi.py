import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE ESTILO ---
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "font.size": 11,
})

# --- CARGAR DATOS ---
print("Cargando datos...")
datos = np.load('datos_tfg_gaugeLandau_g=0_E=0.5.npz')

# Extraemos las variables con los mismos nombres que usamos al guardar
psi = datos['psi']
x = datos['x']
y = datos['y']
Lx = datos['Lx']
Ly = datos['Ly']
N = int(datos['N']) # A veces se carga como float, forzamos int
B = datos['B']

# Reconstruimos las mallas y diferenciales necesarios para derivadas
X, Y = np.meshgrid(x, y)
dx = x[1] - x[0]
dy = y[1] - y[0]
KX, KY = np.meshgrid(np.fft.fftfreq(N, d=dx)*2*np.pi, 
                     np.fft.fftfreq(N, d=dy)*2*np.pi)

print(f"Datos cargados. Grid {N}x{N}. Norma: {np.sum(np.abs(psi)**2)*dx*dy:.4f}")

# REPRESENTACIÓN GRÁFICA
# fig, ax = plt.subplots(figsize=(3.4, 3.0), constrained_layout=True)
# cmap = plt.cm.inferno 
# im = ax.imshow(np.abs(psi[0])**2, cmap=cmap, origin='lower', 
#                extent=[-Lx, Lx, -Ly, Ly], interpolation='bilinear')
# ax.set_xlabel(r'$x/l_B$')
# ax.set_ylabel(r'$y/l_B$')
# cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
# cbar.set_label(r'$|\psi|^2$', rotation=270, labelpad=15)

# plt.savefig('figura_final_desde_archivo.pdf')
# plt.show()

# Definimos el gauge
B = 1
E = 0
A_x = 0
A_y = B * X 
A_t= E*X 


# CALCULO CORRIENTES DE PROBABILIDAD

def Calculo_Corrientes(psi, A_x, A_y, KX, KY, q = 1, hbar = 1, m = 1):
    # Transformamos al espacio de momentos
    psi_k  = np.fft.fft2(psi, axes = (1, 2))
    # Hallamos las derivadas con respecto a x e y \vec p psi
    kx_psi_k = KX*psi_k
    ky_psi_k = KY*psi_k
    #Transformamos de vuelta al espacio de posiciones
    kx_psi = np.fft.ifft2(kx_psi_k, axes=(1, 2))
    ky_psi = np.fft.ifft2(ky_psi_k, axes=(1, 2))
    
    psi_up = psi[0, :, :]
    psi_down = psi[1, :, :]
    psi_up_conj = np.conj(psi_up)
    psi_down_conj = np.conj(psi_down)
    
    # Hallamos el término orbital J_orb = psi^dagger (p - qA) psi
    J_orb_x = (psi_up_conj*(kx_psi[0, :, :] -q*A_x*psi_up) + psi_down_conj*(kx_psi[1, :, :] -q*A_x*psi_down)).real / m
    J_orb_y = (psi_up_conj*(ky_psi[0, :, :] -q*A_y*psi_up) + psi_down_conj*(ky_psi[1, :, :] -q*A_y*psi_down)).real / m
    
    #Hallamos el término de spin J_spin = hbar / 2m * rot(psi^dagger \vec sigma psi)
    # Donde C = psi^dagger \vec sigma psi
    psi_up_conj_psi_down = psi_up_conj  * psi_down
    C_x = 2 * psi_up_conj_psi_down.real
    C_y = 2 * psi_up_conj_psi_down.imag
    C_z = np.abs(psi_up)**2 - np.abs(psi_down)**2
    
    #Transformamos C al espacio de fourier, porque tenemos que aplicarle el rotacional
    C_x_k = np.fft.fft2(C_x)
    C_y_k = np.fft.fft2(C_y)
    C_z_k = np.fft.fft2(C_z)
    
    J_spin_x_k =  1.j * KY * C_z_k
    J_spin_x = hbar / (2*m) * np.fft.ifft2(J_spin_x_k).real
    
    J_spin_y_k =  -1.j * KX * C_z_k
    J_spin_y = hbar / (2*m) * np.fft.ifft2(J_spin_y_k).real
    
    # La componente Z queda pendiente de programar, vamos a ignorarla ahora mismo
    
    return J_orb_x, J_orb_y, J_spin_x, J_spin_y

J_orb_x, J_orb_y, J_spin_x, J_spin_y = Calculo_Corrientes(psi, A_x, A_y, KX, KY)
J_total_x = J_orb_x + J_spin_x
J_total_y = J_orb_y + J_spin_y

J_max = np.max(np.sqrt(J_orb_x**2 + J_orb_y**2))

# --- 2. PLOTTEO DE 3 PANELES ---

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
# densidad
rho = np.abs(psi[0])**2 + np.abs(psi[1])**2
#Definimos la función para representar las corrientes
def plot_stream_panel(ax, Jx, Jy, titulo, color_stream):
    # Fondo: Densidad de probabilidad
    im = ax.imshow(rho, extent =[-Lx, Lx, -Ly, Ly], origin = 'lower', cmap = 'inferno')
    
    # Líneas de corriente(Streamplot)
    # density contronla la cantidad de líneas, linewidth el grosor.
    #arrowsize el tamaño de las flechas

    #vamos a configurar el plot de tal manera que cuanto más intensa sea la corriente
    # más gruesas sean las líneas

    # Calculamos la magnitud del vector en cada punto
    J_mag = np.sqrt(Jx**2 + Jy**2)
    # Normalizamos para que el grosor vaya de 0.5 a 2.5 puntos
    lw = 2.5 * (J_mag / J_max)

    # --- EL TRUCO PARA BORRAR FLECHAS ---
    # Definimos un umbral (por ejemplo, el 5% de la corriente máxima)
    umbral = 0.03 * J_max
    
    # Sustituimos por NaN donde la corriente sea menor al umbral
    Jx_plot = np.where(J_mag > umbral, Jx, np.nan)
    Jy_plot = np.where(J_mag > umbral, Jy, np.nan)
    
    st = ax.streamplot(X, Y, Jx_plot, Jy_plot, color=color_stream, linewidth=lw, density=1.0, arrowsize=1.2)
    ax.set_title(titulo)
    ax.set_xlabel(r'$x/l_B$')
    ax.set_ylabel(r'$y/l_B$')
    
    # --- AÑADE ESTO PARA EVITAR CORTES BLANCOS ---
    ax.set_xlim(-Lx, Lx)
    ax.set_ylim(-Ly, Ly)
    
    return im
    
# Panel 1 - Orbital
plot_stream_panel(axes[0], J_orb_x, J_orb_y, r'Orbital Current $\vec{J}_{orb}$', 'cyan')

# Panel 2 - Spin
plot_stream_panel(axes[1], J_spin_x, J_spin_y, r'Spin Current $\vec{J}_{spin}$', 'white')

# Panel 3 - Total
im = plot_stream_panel(axes[2], J_total_x, J_total_y, r'Total Current $\vec{J}_{tot}$', 'lime')

# Barra de color común (basada en la densidad)
cbar = fig.colorbar(im, ax = axes, shrink = 0.8, fraction=0.05, pad=0.02)
cbar.set_label(r'$|\psi|^2$', rotation=270, labelpad=15)

# Guardar
plt.savefig('fig_corrientes_tfg.png')
plt.show()

# Representación corte transversal del las corrientes de spin $un y fijo; variando x$

# Tomamos el índice correspondiente a y=0 (la mitad de la malla)
idx_y = N // 2

fig1, ax1d = plt.subplots(figsize=(3.35, 2.25))  # ~\columnwidth en muchos journals
plt.tick_params(direction='in', top=True, right=True)

ax1d.plot(x, J_spin_y[idx_y, :] / J_max, label=r"$J_{\mathrm{spin}}$")
ax1d.plot(x, J_orb_y[idx_y, :] / J_max,  label=r"$J_{\mathrm{orb}}$")
ax1d.plot(x, J_total_y[idx_y, :] / J_max,  label=r"$J_{\mathrm{tot}}$")


ax1d.set_xlabel(r'$x/l_B$')
ax1d.set_ylabel(r'$J_y / J_{max}$')

# Zoom solo en la zona central interesante
ax1d.set_xlim(-Lx/1.5, Lx/1.5)

ax1d.legend(loc="upper center", ncol=3, frameon=False,
          bbox_to_anchor=(0.5, 1.18), handlelength=1.6)

fig1.tight_layout(pad=0.15)

#Guardar y mostrar
plt.savefig('fig_corrientes_1D_g50_E0.png')
plt.show()
