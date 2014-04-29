# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# FFT with Python

# <markdowncell>

# If you want to know how the FFT Algorithm works, Jake Vanderplas explained it extremely well in his blog: http://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/

# <markdowncell>

# Here is, how it is applied and how the axis are scaled to real physical values.

# <codecell>

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# <codecell>

%pylab inline --no-import-all

# <headingcell level=2>

# First: A synthetic Signal, a simple Sine Wave

# <codecell>

t = np.linspace(0, 2*np.pi, 1000, endpoint=True)
f = 3.0 # Frequency in Hz
A = 100.0 # Amplitude in Unit
s = A * np.sin(2*np.pi*f*t) # Signal

# <codecell>

plt.plot(t,s)
plt.xlabel('Time ($s$)')
plt.ylabel('Amplitude ($Unit$)')

# <headingcell level=2>

# Do the Discrete Fourier Transform with the Blazing Fast FFT Algorithm

# <codecell>

Y = np.fft.fft(s)

# <markdowncell>

# That's it.

# <headingcell level=3>

# Let's take a look at the result

# <codecell>

plt.plot(Y)

# <markdowncell>

# Hm, looks strange. Something, which is mirrored at the half, right?!

# <codecell>

N = len(Y)/2+1
Y[N-4:N+3]

# <markdowncell>

# Can you see it?
# 
# And it is something with imaginary parts (the $j$) in it. So let's just take the real part of it with the `abs` command.

# <codecell>

plt.plot(np.abs(Y))

# <markdowncell>

# Again, it is perfectly mirrored at the half. So let's just take the first half.

# <headingcell level=2>

# Amplitude Spectrum

# <markdowncell>

# Remember: $N$ is half the length of the output of the FFT.

# <codecell>

plt.plot(np.abs(Y[:N]))

# <markdowncell>

# That looks pretty good. It is called the **amplitude spectrum** of the time domain signal and was calculated with the Discrete Fourier Transform with the *Chuck-Norris-Fast* FFT algorithm. But how to get the x- and y-axis to real physical scaled values?!

# <headingcell level=2>

# Real Physical Values for the Amplitude and Frequency Axes of the FFT

# <headingcell level=3>

# x-Axis: The Frequency Axis of the FFT

# <markdowncell>

# First, let us determine the timestep, which is used to sample the signal. We made it synthetically, but a real signal has a period (measured every second or every day or something similar). If there is no constant frequency, the FFT can not be used! One can interpolate the signal to a new time base, but then the signal spectrum is not the original one. It depends on the case, if the quality is enough or if the information is getting lost with this shift keying. Enough.
# 
# We have a good signal:

# <codecell>

dt = t[1] - t[0]
fa = 1.0/dt # scan frequency
print('dt=%.5fs (Sample Time)' % dt)
print('fa=%.2fHz (Frequency)' % fa)

# <markdowncell>

# Now we need to create a x-Axis vector, which starts from $0.0$ and is filled with $N$ (length of half of the FFT signal) values and going all the way to the maximum frequency, which can be reconstructed. This frequency is half of the maximum sampling frequency ($f_a$) and is called the `Nyquist-Frequency` (see [Nyquist-Shannon Sampling Theorem](http://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem)).

# <codecell>

X = np.linspace(0, fa/2, N, endpoint=True)
X[:4]

# <markdowncell>

# Now let's plot the amplitude spectrum over the newly created frequency vector $X$

# <codecell>

plt.plot(X, np.abs(Y[:N]))
plt.xlabel('Frequency ($Hz$)')

# <markdowncell>

# Yeah! The x-Axis is showing us, that we have a peak at exactly these frequencies, from which our synthetically created signal was build of. That was the job.
# 
# The sample frequency was $f_a=159Hz$, so the amplitude spectrum is from $0.0$ to $\frac{f_a}{2}=79.5Hz$.

# <headingcell level=3>

# y-Axis: The Amplitude of the FFT Signal

# <markdowncell>

# This task is not this easy, because one have to understand, how the Fourier Transform or the Discrete Fourier Transform works in detail. We need to transform the y-axis value from *something* to a real physical value. Because the power of the signal in time and frequency domain have to be equal, and we just used the left half of the signal (look at $N$), now we need to multiply the amplitude with the factor of **2**. If we inverse the FFT with `IFFT`, the power of the signal is the same.
# 
# But that was the easy part. The more complicated one is, if you look at the definition of the Discrete Fourier Transform:
# 
# $Y[k]=\frac{1}{N} \underbrace{\sum_{N} x(nT)\cdot e^{-i 2 \pi \frac{k}{N}n}}_{DFT}$
# 
# In most implementations, the output $Y$ of the `FFT` is normalized with the number of samples. We have to divide by $N$ to get the real physical value.
# 
# The magic factor is $\frac{2}{N}$.

# <codecell>

plt.plot(X, 2.0*np.abs(Y[:N])/N)
plt.xlabel('Frequency ($Hz$)')
plt.ylabel('Amplitude ($Unit$)')

# <markdowncell>

# Yeah! Job accomplised. Congratulations. But wait...
# 
# If you look at the parameters for the original signal ($A$), our signal amplitude was not, what is calculated here. Why??

# <headingcell level=2>

# The wrong Amplitude Spectrum because of Leakage Effect

# <markdowncell>

# Take a look at the original signal.

# <codecell>

plt.plot(t,s)
plt.xlabel('Time ($s$)')
plt.ylabel('Amplitude ($Unit$)')

# <markdowncell>

# Do you see, that the signal do not end at amplitude zero, where it started? That means, if you add these signals up, it looks like this:

# <codecell>

plt.plot(t, s, label='Signal 1')
plt.plot(t+t[-1], s, label='Signal 1 again')
plt.xlim(t[-1]-1, t[-1]+1)
plt.xlabel('Time ($s$)')
plt.ylabel('Amplitude ($Unit$)')
plt.legend()

# <markdowncell>

# And the Fourier Transform was originally invented by Mr Fourier for, and only for, periodic signals (see [Fourier Transform](http://en.wikipedia.org/wiki/Fourier_transform)). So the Discrete Fourier Transform does and the Fast Fourier Transform Algorithm does it, too.
# 
# The signal has to be strictly periodic, which introduces the so called **windowing** to eliminate the leakage effect.

# <headingcell level=2>

# Window Functions to get periodic signals from real data

# <markdowncell>

# There are a lot of window functions, like the *Hamming*, *Hanning*, *Blackman*, ...

# <codecell>

hann = np.hanning(len(s))
hamm = np.hamming(len(s))
black= np.blackman(len(s))

plt.figure(figsize=(8,3))
plt.subplot(131)
plt.plot(hann)
plt.title('Hanning')
plt.subplot(132)
plt.plot(hamm)
plt.title('Hamming')
plt.subplot(133)
plt.plot(black)
plt.title('Blackman')
plt.tight_layout()

# <markdowncell>

# All have different characteristics, which is an [own engineering discipline](http://en.wikipedia.org/wiki/Window_function). Let's take the *Hanning* window function to multiply our signal with.

# <codecell>

plt.plot(t,hann*s)
plt.xlabel('Time ($s$)')
plt.ylabel('Amplitude ($Unit$)')
plt.title('Signal with Hanning Window function applied')

# <headingcell level=2>

# FFT with windowed signal

# <codecell>

Yhann = np.fft.fft(hann*s)

plt.figure(figsize=(7,3))
plt.subplot(121)
plt.plot(t,s)
plt.title('Time Domain Signal')
plt.ylim(np.min(s)*3, np.max(s)*3)
plt.xlabel('Time ($s$)')
plt.ylabel('Amplitude ($Unit$)')

plt.subplot(122)
plt.plot(X, 2.0*np.abs(Yhann[:N])/N)
plt.title('Frequency Domain Signal')
plt.xlabel('Frequency ($Hz$)')
plt.ylabel('Amplitude ($Unit$)')

plt.annotate("FFT",
            xy=(0.0, 0.1), xycoords='axes fraction',
            xytext=(-0.8, 0.2), textcoords='axes fraction',
            size=30, va="center", ha="center",
            arrowprops=dict(arrowstyle="simple",
                            connectionstyle="arc3,rad=0.2"))
plt.tight_layout()

plt.savefig('FFT.png',bbox_inches='tight', dpi=150, transparent=True)

# <markdowncell>

# This is exactly, what we wanted to see: A beautiful amplitude spectrum of our signal, which was calcualted with the FFT algorithm.
# 
# Now let's take a look at some real data!

# <headingcell level=1>

# Vertical Grid Load of Germany 2013

# <markdowncell>

# "The vertical grid load is the sum, positive or negative, of all power transferred from the transmission grid through directly connected transformers and power lines to distribution grids and final consumers."
# 
# Download the Data from [50Hertz.com](http://www.50hertz.com/de/1987.htm)

# <codecell>

!wget -O 'Vertikale_Netzlast_2013.csv' 'http://www.50hertz.com/transmission/files/sync/Netzkennzahlen/Netzlast/ArchivCSV/Vertikale_Netzlast_2013.csv'

# <codecell>

df = pd.read_csv('Vertikale_Netzlast_2013.csv', header=6, sep=';', parse_dates=[[0, 1]], index_col=0, na_values=['n.v.'])
df.rename(columns={'Unnamed: 3': 'Load'}, inplace=True)

# <markdowncell>

# Interpolate the missing data

# <codecell>

df.Load = df.Load.interpolate()

# <codecell>

plt.figure(figsize=(14,5))
df.Load.plot()
plt.title('Vertical Grid Load Germany 2013')
plt.ylabel('Power [$MW$]')
plt.savefig('VerticalGridLoadGermany2013.png',bbox_inches='tight', dpi=150, transparent=True)

# <headingcell level=3>

# Do the FFT

# <codecell>

hann = np.hanning(len(df.Load.values))

# <codecell>

Y = np.fft.fft(hann*df.Load.values)

# <codecell>

N = len(Y)/2+1
fa = 1.0/(15.0*60.0) # every 15 minutes
print('fa=%.4fHz (Frequency)' % fa)

# <codecell>

X = np.linspace(0, fa/2, N, endpoint=True)

# <codecell>

plt.plot(X, 2.0*np.abs(Y[:N])/N)
plt.xlabel('Frequency ($Hz$)')
plt.ylabel('vertical powergrid load ($MW$)')

# <markdowncell>

# Hm. This is not what we expected. For humans, the x-axis is not understandable. What is $0.0002Hz$? Let's convert it to period, which is the reciprocal of the sampling rate.

# <headingcell level=2>

# The Rythm of modern Life, seen in the Power Grid

# <codecell>

Xp = 1.0/X # in seconds
Xph= Xp/(60.0*60.0) # in hours

# <codecell>

plt.figure(figsize=(15,6))
plt.plot(Xph, 2.0*np.abs(Y[:N])/N)
plt.xticks([12, 24, 33, 84, 168])
plt.xlim(0, 180)
plt.ylim(0, 1500)
plt.xlabel('Period ($h$)')
plt.savefig('VerticalGridLoadGermany2013-FFT.png',bbox_inches='tight', dpi=150, transparent=True)

# <markdowncell>

# Aaaaah! Now we see following peaks:
# 
# * `12h` day/night rythm
# * `24h` daily rythm
# * `33.6h` something? Any suggestions?
# * `84.2h` something? Any suggestions?
# * `168h` week rythm

