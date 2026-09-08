import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from IPython.display import Audio, display
from google.colab import files

plt.rcParams["figure.figsize"] = (12, 5)

def calculate_spectrum(x, fs):
    x = np.asarray(x)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    N = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(N, 1/fs)
    A = np.abs(X) / N
    if N > 1:
        A[1:-1] *= 2
    return f, A

def plot_spectrum(x, fs, title="", xlim=None):
    f, A = calculate_spectrum(x, fs)
    plt.figure()
    plt.plot(f, A)
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid()
    if xlim is not None:
        plt.xlim(xlim)
    plt.show()

def normalize_audio(x):
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        x = x.astype(np.float64) / max(abs(info.min), info.max)
    else:
        x = x.astype(np.float64)
        m = np.max(np.abs(x))
        if m > 1:
            x /= m
    return x

def to_mono(x):
    x = np.asarray(x)
    return np.mean(x, axis=1) if x.ndim > 1 else x

def downsample_signal(x, M):
    return np.asarray(x)[::M]

def upsample_signal(x, L):
    x = np.asarray(x)
    y = np.zeros(len(x) * L, dtype=x.dtype)
    y[::L] = x
    return y

def sinc_reconstruction(xn, fs, t):
    n = np.arange(len(xn))
    return np.sum(xn[:, None] * np.sinc(fs*t[None, :] - n[:, None]), axis=0)

def rectangular_reconstruction(xn, fs, t):
    k = np.floor(fs*t).astype(int)
    y = np.zeros_like(t, dtype=float)
    valid = (k >= 0) & (k < len(xn))
    y[valid] = xn[k[valid]]
    return y

def load_audio(name):
    fs, x = wavfile.read(name)
    return fs, normalize_audio(to_mono(x))
fs = 44_100
frequencias = [500, 5_000, 10_000, 50_000]
t = np.arange(0, 0.1, 1/fs)

for f0 in frequencias:
    x = np.cos(2*np.pi*f0*t)
    plot_spectrum(x, fs, f"x(t) = cos(2π·{f0}t)", (0, fs/2))
fs = 44_100
duracao = 2
t = np.arange(0, duracao, 1/fs)
f0 = 500
f1 = 10_000

chirp_linear = signal.chirp(t, f0=f0, f1=f1, t1=duracao, method="linear")
chirp_quadratico = signal.chirp(t, f0=f0, f1=f1, t1=duracao, method="quadratic")
chirp_logaritmico = signal.chirp(t, f0=f0, f1=f1, t1=duracao, method="logarithmic")

for x, nome in [
    (chirp_linear, "Chirp linear"),
    (chirp_quadratico, "Chirp quadrático"),
    (chirp_logaritmico, "Chirp logarítmico")
]:
    plot_spectrum(x, fs, nome, (0, 12_000))
fs_handel, handel = load_audio("handel.wav")
print(f"Frequência de amostragem: {fs_handel} Hz")
print(f"Número de amostras: {len(handel)}")
print(f"Duração: {len(handel)/fs_handel:.2f} s")

plot_spectrum(handel, fs_handel, "Espectro de handel.wav", (0, fs_handel/2))
display(Audio(handel, rate=fs_handel))
M_values = [2, 4, 8]
handel_down = {}

for M in M_values:
    y = downsample_signal(handel, M)
    fs_y = fs_handel/M
    handel_down[M] = (y, fs_y)
    print(f"M = {M} | fs = {fs_y:.1f} Hz | duração = {len(y)/fs_y:.2f} s")
    plot_spectrum(y, fs_y, f"Subamostragem por M = {M}", (0, fs_y/2))
    display(Audio(y, rate=int(fs_y)))
handel_resample_down = {}

for M in M_values:
    y = signal.resample(handel, len(handel)//M)
    fs_y = fs_handel/M
    handel_resample_down[M] = (y, fs_y)
    print(f"M = {M} | fs = {fs_y:.1f} Hz | duração = {len(y)/fs_y:.2f} s")
    plot_spectrum(y, fs_y, f"scipy.signal.resample() — M = {M}", (0, fs_y/2))
    display(Audio(y, rate=int(fs_y)))
handel_up = {}

for L in M_values:
    y = upsample_signal(handel, L)
    fs_y = fs_handel*L
    handel_up[L] = (y, fs_y)
    print(f"L = {L} | fs = {fs_y} Hz | duração = {len(y)/fs_y:.2f} s")
    plot_spectrum(y, fs_y, f"Sobreamostragem por inserção de zeros — L = {L}", (0, fs_y/2))
    display(Audio(y, rate=int(fs_y)))
handel_resample_up = {}

for L in M_values:
    y = signal.resample(handel, len(handel)*L)
    fs_y = fs_handel*L
    handel_resample_up[L] = (y, fs_y)
    print(f"L = {L} | fs = {fs_y} Hz | duração = {len(y)/fs_y:.2f} s")
    plot_spectrum(y, fs_y, f"scipy.signal.resample() — L = {L}", (0, fs_y/2))
    display(Audio(y, rate=int(fs_y)))
fs_cont = 10_000_000
duracao = 0.01
t_cont = np.arange(0, duracao, 1/fs_cont)
x_cont = np.cos(2000*np.pi*t_cont) + np.sin(5000*np.pi*t_cont)

plt.figure()
plt.plot(t_cont*1000, x_cont)
plt.xlabel("Tempo (ms)")
plt.ylabel("Amplitude")
plt.title("x(t) no domínio do tempo")
plt.grid()
plt.show()

plot_spectrum(x_cont, fs_cont, "Espectro de x(t)", (0, 5_000))
fs_nyquist = 5_000
n = np.arange(int(duracao*fs_nyquist))
t_n = n/fs_nyquist
x_n = np.cos(2000*np.pi*t_n) + np.sin(5000*np.pi*t_n)

print(f"Frequência máxima: 2500 Hz")
print(f"Frequência de Nyquist utilizada: {fs_nyquist} Hz")

plt.figure()
plt.stem(t_n*1000, x_n)
plt.xlabel("Tempo (ms)")
plt.ylabel("Amplitude")
plt.title("x[n] amostrado na frequência de Nyquist")
plt.grid()
plt.show()

plot_spectrum(x_n, fs_nyquist, "Espectro de x[n]", (0, fs_nyquist/2))
t_rec = np.linspace(0, duracao, 20_000, endpoint=False)

x_sinc = sinc_reconstruction(x_n, fs_nyquist, t_rec)
x_rect = rectangular_reconstruction(x_n, fs_nyquist, t_rec)

plt.figure()
plt.plot(t_rec*1000, x_sinc)
plt.xlabel("Tempo (ms)")
plt.ylabel("Amplitude")
plt.title("Reconstrução por interpolação sinc")
plt.grid()
plt.show()

plt.figure()
plt.plot(t_rec*1000, x_rect)
plt.xlabel("Tempo (ms)")
plt.ylabel("Amplitude")
plt.title("Reconstrução por pulsos retangulares")
plt.grid()
plt.show()

plt.figure()
plt.plot(t_cont*1000, x_cont, label="Original")
plt.plot(t_rec*1000, x_sinc, label="Sinc")
plt.xlabel("Tempo (ms)")
plt.ylabel("Amplitude")
plt.title("Sinal original e reconstrução sinc")
plt.legend()
plt.grid()
plt.show()
fs_hbanheiro, hbanheiro = load_audio("h_banheiro.wav")
fs_taca, taca = load_audio("sinal_taca.wav")

print(f"h_banheiro.wav: {fs_hbanheiro} Hz")
print(f"sinal_taca.wav: {fs_taca} Hz")

plot_spectrum(hbanheiro, fs_hbanheiro, "Espectro de h_banheiro.wav", (0, fs_hbanheiro/2))
plot_spectrum(taca, fs_taca, "Espectro de sinal_taca.wav", (0, fs_taca/2))

display(Audio(hbanheiro, rate=fs_hbanheiro))
display(Audio(taca, rate=fs_taca))

from scipy.signal import resample

# Reamostragem para igualar as frequências de amostragem
if fs_hbanheiro != fs_handel:
    num_amostras = int(len(handel) * fs_hbanheiro / fs_handel)
    handel = resample(handel, num_amostras)
    fs_handel = fs_hbanheiro

if fs_taca != fs_hbanheiro:
    num_amostras = int(len(taca) * fs_hbanheiro / fs_taca)
    taca = resample(taca, num_amostras)
    fs_taca = fs_hbanheiro

resposta_handel = signal.fftconvolve(handel, hbanheiro, mode="full")
resposta_taca = signal.fftconvolve(taca, hbanheiro, mode="full")

fs_respostas = fs_hbanheiro

resposta_handel /= max(np.max(np.abs(resposta_handel)), 1)
resposta_taca /= max(np.max(np.abs(resposta_taca)), 1)

plot_spectrum(resposta_handel, fs_respostas, "Resposta de hbanheiro[n] ao handel.wav", (0, fs_respostas/2))
plot_spectrum(resposta_taca, fs_respostas, "Resposta de hbanheiro[n] ao sinal_taca.wav", (0, fs_respostas/2))

display(Audio(resposta_handel, rate=fs_respostas))
display(Audio(resposta_taca, rate=fs_respostas))
