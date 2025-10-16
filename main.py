import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.fft import fft, ifft
from scipy import signal
import wx

AUDIO_IR = 'sounds/Banheiro.wav'
BLOCK_SIZE = 2048
MIX_DRY = 0.7
MIX_WET = 0.3

# Leitura da resposta ao impulso
h, sr_h = sf.read(AUDIO_IR)
if h.ndim > 1:
    h = h.mean(axis=1)
h = h.astype(np.float32)
h /= np.max(np.abs(h) + 1e-8)

SR = int(sr_h)
print(f"Taxa de amostragem: {SR} Hz")

# Preparação da convolução
N = BLOCK_SIZE + len(h) - 1
Nfft = 1 << (N - 1).bit_length()
H = fft(h, Nfft)
overlap = np.zeros(len(h) - 1, dtype=np.float32)

stream = None
reverbe_ativo = True
high_pass_ativo = False
low_pass_ativo = False

# Criação dos filtros Butterworth
b_high, a_high = signal.butter(4, 300 / (SR / 2), btype='high')
b_low, a_low = signal.butter(4, 6000 / (SR / 2), btype='low')

# Estados iniciais dos filtros
zi_high = signal.lfilter_zi(b_high, a_high)
zi_low = signal.lfilter_zi(b_low, a_low)
state_high = None
state_low = None


def audio_callback(indata, outdata, frames, time, status):
    global overlap, reverbe_ativo, high_pass_ativo, low_pass_ativo
    global state_high, state_low

    if status:
        print(status)

    x = indata[:, 0]

    if reverbe_ativo:
        X = fft(x, Nfft)
        Y = X * H
        y = np.real(ifft(Y))[:len(x) + len(h) - 1]
        y[:len(overlap)] += overlap
        overlap = y[len(x):]
        y_conv = y[:len(x)]
        y_out = MIX_DRY * x + MIX_WET * y_conv
    else:
        y_out = x.copy()

    if high_pass_ativo:
        if state_high is None:
            y_out, state_high = signal.lfilter(b_high, a_high, y_out, zi=zi_high * y_out[0])
        else:
            y_out, state_high = signal.lfilter(b_high, a_high, y_out, zi=state_high)
    if low_pass_ativo:
        if state_low is None:
            y_out, state_low = signal.lfilter(b_low, a_low, y_out, zi=zi_low * y_out[0])
        else:
            y_out, state_low = signal.lfilter(b_low, a_low, y_out, zi=state_low)

    y_out = np.clip(y_out, -1.0, 1.0)
    outdata[:, 0] = y_out


class AudioApp(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Processamento de Áudio", size=(320, 230))
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        titulo = wx.StaticText(panel, label="Processamento de Áudio em Tempo Real")
        fonte = titulo.GetFont()
        fonte.PointSize += 2
        fonte.MakeBold()
        titulo.SetFont(fonte)
        vbox.Add(titulo, flag=wx.ALL | wx.ALIGN_CENTER, border=10)

        self.botao_mic = wx.Button(panel, label="Ativar Microfone")
        self.botao_mic.Bind(wx.EVT_BUTTON, self.toggle_microfone)
        vbox.Add(self.botao_mic, flag=wx.ALL | wx.EXPAND, border=10)

        self.check_high = wx.CheckBox(panel, label="High-Pass")
        self.check_high.Bind(wx.EVT_CHECKBOX, self.toggle_high)
        vbox.Add(self.check_high, flag=wx.ALL, border=5)

        self.check_low = wx.CheckBox(panel, label="Low-Pass")
        self.check_low.Bind(wx.EVT_CHECKBOX, self.toggle_low)
        vbox.Add(self.check_low, flag=wx.ALL, border=5)

        self.check_reverbe = wx.CheckBox(panel, label="Convolução")
        self.check_reverbe.SetValue(True)
        self.check_reverbe.Bind(wx.EVT_CHECKBOX, self.toggle_reverbe)
        vbox.Add(self.check_reverbe, flag=wx.ALL, border=5)

        panel.SetSizer(vbox)
        self.Centre()
        self.Show()

    def toggle_microfone(self, event):
        global stream, overlap, state_high, state_low
        if stream is None:
            overlap[:] = 0
            state_high = None
            state_low = None
            stream = sd.Stream(channels=1, samplerate=SR, blocksize=BLOCK_SIZE,
                               dtype='float32', callback=audio_callback)
            stream.start()
            self.botao_mic.SetLabel("Desativar Microfone")
        else:
            stream.stop()
            stream.close()
            stream = None
            overlap[:] = 0
            state_high = None
            state_low = None
            self.botao_mic.SetLabel("Ativar Microfone")

    def toggle_reverbe(self, event):
        global reverbe_ativo
        reverbe_ativo = self.check_reverbe.GetValue()

    def toggle_high(self, event):
        global high_pass_ativo
        high_pass_ativo = self.check_high.GetValue()

    def toggle_low(self, event):
        global low_pass_ativo
        low_pass_ativo = self.check_low.GetValue()


if __name__ == "__main__":
    app = wx.App()
    AudioApp()
    app.MainLoop()
