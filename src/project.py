import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import IPython
import statistics
import wavio
from scipy.io import wavfile
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk

# Priklad 1
#=====================================================================
fs, data = wavfile.read('./audio/xvanoj00.wav')
data.min(), data.max()
print('Pr1:')
print('pocet vzorkov:',len(data))
print('dlzka v sekundach:',len(data)/fs)
print('max:',max(data))
print('min:',min(data))
print('------------------------')
data = data[:len(data)]
t = np.arange(data.size) / fs


plt.figure(figsize= (12,6))
plt.plot(t, data)
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_title('Zvukovy signal')
plt.tight_layout()
plt.savefig('pr1.png')
#=====================================================================

# Priklad 2
#=====================================================================
median = statistics.median((data))
tmp = 0
for i in data:
    data[tmp] = i - median
    tmp += 1
data = data / max(data)

ramce = []
num = round(len(data)/512)
c = 1
a = 0
b = 1024
for i in range(num):
    ramec = data[a:b]
    a += 512
    b += 512
    plt.figure(figsize= (12,6))
    plt.plot(np.arange(ramec.size)/fs, ramec)
    plt.gca().set_xlabel('$t[s]$')
    plt.gca().set_title('Ramec:'+ str(c))
    plt.tight_layout()
    plt.savefig(str(c)+'.png')
    plt.close()
    c += 1
    ramce.append(ramec)   
#=====================================================================

# Priklad 3
#=====================================================================
def DFT(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    
    dft = np.dot(e, x)
    
    return np.array(dft, dtype='complex_')


dft = DFT(ramce[53])
dft_left = dft[:len(dft)//2]
plt.figure(figsize= (12,6))
plt.plot(np.arange(dft_left.size)*fs/1024, np.abs(dft_left))
plt.gca().set_xlabel('$f[Hz]$')
plt.gca().set_title('Moje DFT')
plt.tight_layout()
plt.savefig('dft.png')
plt.close()

np_dft = np.fft.fft(ramce[53], 1024)
np_dft_left = np_dft[:len(np_dft)//2]
plt.figure(figsize=(12,6))
plt.plot(np.arange(np_dft_left.size)*fs/1024, np.abs(np_dft_left))
plt.gca().set_xlabel('$f [Hz]$')
plt.gca().set_title('Numpy DFT')
plt.tight_layout()
plt.savefig('np_dft.png')
plt.close()
#=====================================================================

# Priklad 4 
#=====================================================================
f, t, sgr = spectrogram(data, fs, nperseg=1024, noverlap=512)
sgr_log = 10 * np.log10(sgr+1e-20)

plt.figure(figsize=(9,6))
plt.pcolormesh(t, f, sgr_log)
plt.gca().set_xlabel('Cas [s]')
plt.gca().set_ylabel('Frekvencia [Hz]')
cbar = plt.colorbar()
cbar.set_label('Spektralna hustota vykonu [dB]', rotation=270, labelpad=15)
plt.tight_layout()
plt.savefig('Spektrogram.png')
plt.close()
#=====================================================================

# Priklad 5
#=====================================================================
f1 = 585
f2 = 1170
f3 = 1755
f4 = 2340
#=====================================================================

# Priklad 6
#=====================================================================
pole = []
for i in range(len(data)):
    pole.append(i/fs)
    
cos_f1 = np.cos(2 * np.pi * f1 * np.array(pole))
cos_f2 = np.cos(2 * np.pi * f2 * np.array(pole))
cos_f3 = np.cos(2 * np.pi * f3 * np.array(pole))
cos_f4 = np.cos(2 * np.pi * f4 * np.array(pole))

cos = cos_f1 + cos_f2 + cos_f3 + cos_f4
wavfile.write('audio/4cos.wav', int(fs), cos.astype(np.float32))

fs, data = wavfile.read('./audio/4cos.wav')
f, t, sgr = spectrogram(data, fs, nperseg=1024, noverlap=512)
sgr_log = 10 * np.log10(sgr+1e-20)

plt.figure(figsize=(9,3))
plt.pcolormesh(t,f,sgr_log)
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)

plt.tight_layout()

plt.savefig('pr6')
plt.close()
#=====================================================================