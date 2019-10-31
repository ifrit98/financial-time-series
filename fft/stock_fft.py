import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data_og = pd.read_csv('/home/home/Desktop/transfer/financial-time-series/google.csv', thousands=',')
data = data_og.loc[:,["Date", "Close"]] # data_og[['Date', 'Close']]
data['Close'] = data['Close'].apply(lambda x: float(x))
data.head(n=10)

data.plot(y='Close', x='Date')
plt.show()

# Take FFT
close_fft = np.fft.fft(np.asarray(data['Close'].tolist()))
fft_df = pd.DataFrame({'fft':close_fft})


fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
fft_df.head(n=20)


from collections import deque
items = deque(np.asarray(fft_df['absolute'].tolist()))
items.rotate(int(np.floor(len(fft_df)/2)))
plt.figure()
plt.stem(items)
plt.show()


# Reproduce the original signal
plt.figure()
plt.plot(np.fft.ifft(np.asarray(fft_df['fft'].tolist())))
plt.show()


# Low-pass filter (take [200, 100, 50, 20, 10 components)
component_windows = [100, 50, 25, 10, 5]

# Show range of fft components
def plot_components(fft_df, component_windows = [100, 50, 25, 10, 5]):
  fft_list = np.asarray(fft_df['fft'].to_list())
  for i in range(len(component_windows)):
    start = end = component_windows[i]
    fft_list[start:-end] = 0
    plt.figure()
    plt.plot(np.fft.ifft(fft_list))
    plt.show()
    input("Press Enter to continue...")

plot_components(fft_df, component_windows)

