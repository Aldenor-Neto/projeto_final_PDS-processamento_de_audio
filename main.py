import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.fft import fft, ifft
from scipy import signal
import wx
import queue

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

input_stream = None
output_stream = None
audio_queue = queue.Queue(maxsize=5)  # Fila para áudio processado
reverbe_ativo = True
high_pass_ativo = False
low_pass_ativo = False
distortion_ativo = False
tremolo_ativo = False
delay_ativo = False

# Criação dos filtros Butterworth
b_high, a_high = signal.butter(4, 300 / (SR / 2), btype='high')
b_low, a_low = signal.butter(4, 6000 / (SR / 2), btype='low')

# Estados iniciais dos filtros
zi_high = signal.lfilter_zi(b_high, a_high)
zi_low = signal.lfilter_zi(b_low, a_low)
state_high = None
state_low = None

# Parâmetros dos efeitos
dist_gain = 20.0
tremolo_rate = 5.0
tremolo_depth = 0.6
tremolo_phase = 0.0
delay_time = 0.4
delay_feedback = 0.5
delay_buffer_len = int(SR * 2)
delay_buffer = np.zeros(delay_buffer_len, dtype=np.float32)
delay_write_head = 0


def apply_distortion(x, gain):
    """Distorção Hard Clipping"""
    out = x * gain
    out = np.clip(out, -0.6, 0.6)
    return out * 1.2

def apply_tremolo(x, rate, depth, phase, sr):
    """Tremolo (AM Synthesis)"""
    out = np.zeros_like(x)
    two_pi = 2 * np.pi
    phase_increment = two_pi * rate / sr
    current_phase = phase
    
    for i in range(len(x)):
        modulator = 1.0 - depth * (0.5 * (1.0 + np.sin(current_phase)))
        out[i] = x[i] * modulator
        current_phase += phase_increment
        if current_phase > two_pi:
            current_phase -= two_pi
    
    return out, current_phase

def apply_delay(x, buffer, write_ptr, delay_samps, feedback, buf_len):
    """Delay (Eco Simples com Feedback) usando Buffer Circular"""
    out = np.zeros_like(x)
    current_write = write_ptr
    
    for i in range(len(x)):
        read_ptr = current_write - delay_samps
        if read_ptr < 0:
            read_ptr += buf_len
        
        delayed_sample = buffer[read_ptr]
        input_val = x[i]
        out[i] = input_val + delayed_sample
        buffer[current_write] = input_val + (delayed_sample * feedback)
        
        current_write += 1
        if current_write >= buf_len:
            current_write = 0
    
    return out, buffer, current_write

def input_callback(indata, frames, time, status):
    """Callback para captura de áudio - apenas processa, não reproduz"""
    global overlap, reverbe_ativo, high_pass_ativo, low_pass_ativo
    global distortion_ativo, tremolo_ativo, delay_ativo
    global state_high, state_low, audio_queue
    global dist_gain, tremolo_rate, tremolo_depth, tremolo_phase
    global delay_time, delay_feedback, delay_buffer, delay_write_head, delay_buffer_len

    if status:
        print(status)
    
    # Processa apenas o primeiro canal de entrada
    x = indata[:, 0]
    y_out = x.copy()

    # 1. Distorção (primeiro na cadeia)
    if distortion_ativo:
        y_out = apply_distortion(y_out, dist_gain)

    # 2. Filtros
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

    # 3. Tremolo
    if tremolo_ativo:
        y_out, tremolo_phase = apply_tremolo(y_out, tremolo_rate, tremolo_depth, tremolo_phase, SR)

    # 4. Delay
    if delay_ativo:
        delay_samples = int(delay_time * SR)
        y_out, delay_buffer, delay_write_head = apply_delay(
            y_out, delay_buffer, delay_write_head, delay_samples, 
            delay_feedback, delay_buffer_len
        )

    # 5. Reverb (último na cadeia)
    if reverbe_ativo:
        X = fft(y_out, Nfft)
        Y = X * H
        y = np.real(ifft(Y))[:len(y_out) + len(h) - 1]
        y[:len(overlap)] += overlap
        overlap = y[len(y_out):]
        y_conv = y[:len(y_out)]
        y_out = MIX_DRY * y_out + MIX_WET * y_conv

    y_out = np.clip(y_out, -1.0, 1.0)
    # Coloca o áudio processado na fila para reprodução
    try:
        audio_queue.put_nowait(y_out)
    except queue.Full:
        pass

def output_callback(outdata, frames, time, status):
    """Callback para reprodução - apenas reproduz o áudio processado"""
    global audio_queue
    
    if status:
        print(status)
    
    try:
        # Pega o áudio processado da fila
        audio_data = audio_queue.get_nowait()
        outdata[:, 0] = audio_data
    except queue.Empty:
        # Se não houver áudio processado, silencia
        outdata.fill(0)


class AudioApp(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Processamento de Áudio", size=(400, 500))
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

        # Seção Distorção
        box_dist = wx.StaticBox(panel, label="Distorção")
        sizer_dist = wx.StaticBoxSizer(box_dist, wx.VERTICAL)
        self.check_distortion = wx.CheckBox(panel, label="Ativar Distorção")
        self.check_distortion.Bind(wx.EVT_CHECKBOX, self.toggle_distortion)
        sizer_dist.Add(self.check_distortion, flag=wx.ALL, border=5)
        self.slider_dist = wx.Slider(panel, value=20, minValue=1, maxValue=50, style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        self.slider_dist.Bind(wx.EVT_SLIDER, self.update_distortion)
        sizer_dist.Add(self.slider_dist, flag=wx.ALL | wx.EXPAND, border=5)
        vbox.Add(sizer_dist, flag=wx.ALL | wx.EXPAND, border=5)

        # Seção Filtros
        box_filter = wx.StaticBox(panel, label="Filtros")
        sizer_filter = wx.StaticBoxSizer(box_filter, wx.VERTICAL)
        self.check_high = wx.CheckBox(panel, label="High-Pass (300Hz)")
        self.check_high.Bind(wx.EVT_CHECKBOX, self.toggle_high)
        sizer_filter.Add(self.check_high, flag=wx.ALL, border=5)
        self.check_low = wx.CheckBox(panel, label="Low-Pass (6000Hz)")
        self.check_low.Bind(wx.EVT_CHECKBOX, self.toggle_low)
        sizer_filter.Add(self.check_low, flag=wx.ALL, border=5)
        vbox.Add(sizer_filter, flag=wx.ALL | wx.EXPAND, border=5)

        # Seção Tremolo
        box_trem = wx.StaticBox(panel, label="Tremolo")
        sizer_trem = wx.StaticBoxSizer(box_trem, wx.VERTICAL)
        self.check_tremolo = wx.CheckBox(panel, label="Ativar Tremolo")
        self.check_tremolo.Bind(wx.EVT_CHECKBOX, self.toggle_tremolo)
        sizer_trem.Add(self.check_tremolo, flag=wx.ALL, border=5)
        self.slider_trem_rate = wx.Slider(panel, value=5, minValue=1, maxValue=15, style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        self.slider_trem_rate.Bind(wx.EVT_SLIDER, self.update_tremolo)
        sizer_trem.Add(wx.StaticText(panel, label="Velocidade (Hz)"), flag=wx.ALL, border=2)
        sizer_trem.Add(self.slider_trem_rate, flag=wx.ALL | wx.EXPAND, border=5)
        self.slider_trem_depth = wx.Slider(panel, value=60, minValue=0, maxValue=100, style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        self.slider_trem_depth.Bind(wx.EVT_SLIDER, self.update_tremolo)
        sizer_trem.Add(wx.StaticText(panel, label="Profundidade"), flag=wx.ALL, border=2)
        sizer_trem.Add(self.slider_trem_depth, flag=wx.ALL | wx.EXPAND, border=5)
        vbox.Add(sizer_trem, flag=wx.ALL | wx.EXPAND, border=5)

        # Seção Delay
        box_delay = wx.StaticBox(panel, label="Delay (Eco)")
        sizer_delay = wx.StaticBoxSizer(box_delay, wx.VERTICAL)
        self.check_delay = wx.CheckBox(panel, label="Ativar Delay")
        self.check_delay.Bind(wx.EVT_CHECKBOX, self.toggle_delay)
        sizer_delay.Add(self.check_delay, flag=wx.ALL, border=5)
        self.slider_delay_time = wx.Slider(panel, value=40, minValue=10, maxValue=100, style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        self.slider_delay_time.Bind(wx.EVT_SLIDER, self.update_delay)
        sizer_delay.Add(wx.StaticText(panel, label="Tempo (0.1-1.0s)"), flag=wx.ALL, border=2)
        sizer_delay.Add(self.slider_delay_time, flag=wx.ALL | wx.EXPAND, border=5)
        self.slider_delay_fb = wx.Slider(panel, value=50, minValue=0, maxValue=90, style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        self.slider_delay_fb.Bind(wx.EVT_SLIDER, self.update_delay)
        sizer_delay.Add(wx.StaticText(panel, label="Feedback (0-0.9)"), flag=wx.ALL, border=2)
        sizer_delay.Add(self.slider_delay_fb, flag=wx.ALL | wx.EXPAND, border=5)
        vbox.Add(sizer_delay, flag=wx.ALL | wx.EXPAND, border=5)

        # Seção Reverb
        box_reverb = wx.StaticBox(panel, label="Reverb (Convolução)")
        sizer_reverb = wx.StaticBoxSizer(box_reverb, wx.VERTICAL)
        self.check_reverbe = wx.CheckBox(panel, label="Ativar Reverb")
        self.check_reverbe.SetValue(True)
        self.check_reverbe.Bind(wx.EVT_CHECKBOX, self.toggle_reverbe)
        sizer_reverb.Add(self.check_reverbe, flag=wx.ALL, border=5)
        vbox.Add(sizer_reverb, flag=wx.ALL | wx.EXPAND, border=5)

        panel.SetSizer(vbox)
        self.Centre()
        # Inicializa os valores dos parâmetros
        self.update_distortion(None)
        self.update_tremolo(None)
        self.update_delay(None)
        self.Show()

    def toggle_microfone(self, event):
        global input_stream, output_stream, overlap, state_high, state_low, audio_queue
        global delay_buffer, delay_write_head, tremolo_phase
        if input_stream is None:
            overlap[:] = 0
            state_high = None
            state_low = None
            delay_buffer.fill(0)
            delay_write_head = 0
            tremolo_phase = 0.0
            # Limpa a fila
            while not audio_queue.empty():
                try: audio_queue.get_nowait()
                except: break
            # Streams separados: entrada e saída independentes
            # Isso evita o monitoramento automático do Windows
            input_stream = sd.InputStream(
                channels=1,
                samplerate=SR,
                blocksize=BLOCK_SIZE,
                dtype='float32',
                callback=input_callback,
                latency='low'
            )
            output_stream = sd.OutputStream(
                channels=1,
                samplerate=SR,
                blocksize=BLOCK_SIZE,
                dtype='float32',
                callback=output_callback,
                latency='low'
            )
            input_stream.start()
            output_stream.start()
            self.botao_mic.SetLabel("Desativar Microfone")
        else:
            input_stream.stop()
            input_stream.close()
            output_stream.stop()
            output_stream.close()
            input_stream = None
            output_stream = None
            overlap[:] = 0
            state_high = None
            state_low = None
            delay_buffer.fill(0)
            delay_write_head = 0
            tremolo_phase = 0.0
            # Limpa a fila
            while not audio_queue.empty():
                try: audio_queue.get_nowait()
                except: break
            self.botao_mic.SetLabel("Ativar Microfone")

    def toggle_distortion(self, event):
        global distortion_ativo
        distortion_ativo = self.check_distortion.GetValue()

    def update_distortion(self, event):
        global dist_gain
        dist_gain = self.slider_dist.GetValue()

    def toggle_tremolo(self, event):
        global tremolo_ativo
        tremolo_ativo = self.check_tremolo.GetValue()

    def update_tremolo(self, event):
        global tremolo_rate, tremolo_depth
        tremolo_rate = self.slider_trem_rate.GetValue()
        tremolo_depth = self.slider_trem_depth.GetValue() / 100.0

    def toggle_delay(self, event):
        global delay_ativo
        delay_ativo = self.check_delay.GetValue()

    def update_delay(self, event):
        global delay_time, delay_feedback
        delay_time = self.slider_delay_time.GetValue() / 100.0
        delay_feedback = self.slider_delay_fb.GetValue() / 100.0

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
