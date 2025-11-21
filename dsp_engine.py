import numpy as np
import soundfile as sf
from scipy import signal
from scipy.fft import fft, ifft
from numba import jit
import os

class AudioProcessor:
    def __init__(self, block_size=1024, sample_rate=44100, ir_path='sounds/Banheiro.wav'):
        self.block_size = block_size
        self.sr = sample_rate
        
        # --- Parâmetros Gerais ---
        self.mix_dry = 0.7
        self.mix_wet = 0.3
        
        # --- Flags de Efeitos ---
        self.high_pass_on = False
        self.low_pass_on = False
        self.distortion_on = False
        self.tremolo_on = False
        self.delay_on = False
        self.reverb_on = True
        
        # --- Parâmetros dos Efeitos ---
        self.dist_gain = 20.0       # Ganho da distorção (Drive)
        self.tremolo_rate = 5.0     # Hz (Velocidade)
        self.tremolo_depth = 0.6    # Profundidade (0 a 1)
        self.delay_time = 0.4       # Segundos
        self.delay_feedback = 0.5   # Quanto som volta (0 a 1)

        # --- Inicialização de Memória (States) ---
        
        # 1. Reverb (Convolução FFT)
        self._init_reverb(ir_path)
        
        # 2. Filtros (Butterworth)
        self.hp_b, self.hp_a = signal.butter(2, 300 / (self.sr / 2), btype='high')
        self.lp_b, self.lp_a = signal.butter(2, 5000 / (self.sr / 2), btype='low')
        self.hp_state = np.zeros(2, dtype=np.float64) 
        self.lp_state = np.zeros(2, dtype=np.float64)
        
        # 3. Tremolo (Fase do Oscilador)
        self.tremolo_phase = 0.0
        
        # 4. Delay (Buffer Circular)
        # Criamos um buffer de 2 segundos para ter espaço de sobra
        self.delay_buffer_len = int(self.sr * 2) 
        self.delay_buffer = np.zeros(self.delay_buffer_len, dtype=np.float32)
        self.delay_write_head = 0 # Ponteiro de escrita

    def _init_reverb(self, path):
        """Prepara a resposta ao impulso e buffers para Overlap-Add"""
        try:
            if os.path.exists(path):
                h, sr_h = sf.read(path)
                if h.ndim > 1: h = np.mean(h, axis=1)
            else:
                raise FileNotFoundError
        except:
            print("IR não encontrado. Usando sintético.")
            h = np.random.randn(int(self.sr * 1.5)) * np.exp(-np.linspace(0, 10, int(self.sr * 1.5)))

        h = h.astype(np.float32)
        h /= np.max(np.abs(h)) + 1e-8
        self.M = len(h)
        self.L = self.block_size
        self.N_fft = 1 << (self.M + self.L - 1).bit_length()
        self.H_fft = fft(h, self.N_fft)
        self.reverb_overlap = np.zeros(self.N_fft, dtype=np.float32)

    def process_block(self, input_block):
        """Cadeia de Processamento de Efeitos"""
        output = input_block.copy()

        # 1. Distorção (Satura o sinal primeiro)
        if self.distortion_on:
            output = apply_distortion_numba(output, self.dist_gain)

        # 2. Filtros (Equalização)
        if self.high_pass_on:
            output, self.hp_state = apply_iir_filter_numba(output, self.hp_b, self.hp_a, self.hp_state)
        if self.low_pass_on:
            output, self.lp_state = apply_iir_filter_numba(output, self.lp_b, self.lp_a, self.lp_state)

        # 3. Tremolo (Modulação de Amplitude)
        if self.tremolo_on:
            output, self.tremolo_phase = apply_tremolo_numba(
                output, self.tremolo_rate, self.tremolo_depth, self.tremolo_phase, self.sr
            )

        # 4. Delay (Eco/Repetição)
        if self.delay_on:
            delay_samples = int(self.delay_time * self.sr)
            output, self.delay_buffer, self.delay_write_head = apply_delay_numba(
                output, self.delay_buffer, self.delay_write_head, 
                delay_samples, self.delay_feedback, self.delay_buffer_len
            )

        # 5. Reverb (Ambiência Final)
        if self.reverb_on:
            X_fft = fft(output, self.N_fft) # Note que usamos 'output' aqui, não input_block
            Y_fft = X_fft * self.H_fft
            y_time = np.real(ifft(Y_fft))
            y_time += self.reverb_overlap
            y_out = y_time[:self.L]
            self.reverb_overlap = np.zeros(self.N_fft, dtype=np.float32)
            self.reverb_overlap[:(self.N_fft - self.L)] = y_time[self.L:]
            
            # Mixagem Reverb
            output = (output * self.mix_dry) + (y_out * self.mix_wet)

        return np.clip(output, -1.0, 1.0)


# --- ALGORITMOS OTIMIZADOS (NUMBA) ---

@jit(nopython=True)
def apply_iir_filter_numba(x, b, a, state):
    """Filtro Digital IIR (Direct Form II Transposed)"""
    y = np.zeros_like(x)
    b0, b1, b2 = b[0], b[1], b[2]
    a0, a1, a2 = a[0], a[1], a[2]
    s1, s2 = state[0], state[1]
    
    for n in range(len(x)):
        in_val = x[n]
        out_val = b0 * in_val + s1
        s1 = b1 * in_val - a1 * out_val + s2
        s2 = b2 * in_val - a2 * out_val
        y[n] = out_val
        
    return y, np.array([s1, s2], dtype=np.float64)

@jit(nopython=True)
def apply_distortion_numba(x, gain):
    """
    Distorção Hard Clipping
    Multiplica o sinal (Gain) e corta tudo que passa de 1.0 ou -1.0.
    Isso cria ondas quadradas, adicionando muitos harmônicos.
    """
    out = x * gain
    for i in range(len(out)):
        if out[i] > 0.6: # Threshold de corte
            out[i] = 0.6
        elif out[i] < -0.6:
            out[i] = -0.6
    
    # Compensa volume após o corte agressivo
    return out * 1.2 

@jit(nopython=True)
def apply_tremolo_numba(x, rate, depth, phase, sr):
    """
    Tremolo (AM Synthesis)
    Multiplica o sinal por uma onda senoidal de baixa frequência (LFO).
    """
    out = np.zeros_like(x)
    two_pi = 2 * np.pi
    # Quanto a fase avança por amostra
    phase_increment = two_pi * rate / sr
    
    current_phase = phase
    
    for i in range(len(x)):
        # Oscilador varia de (1-depth) até 1
        # Se depth = 1, volume vai a zero. Se depth = 0, volume constante.
        modulator = 1.0 - depth * (0.5 * (1.0 + np.sin(current_phase)))
        
        out[i] = x[i] * modulator
        
        # Avança o oscilador
        current_phase += phase_increment
        if current_phase > two_pi:
            current_phase -= two_pi
            
    return out, current_phase

@jit(nopython=True)
def apply_delay_numba(x, buffer, write_ptr, delay_samps, feedback, buf_len):
    """
    Delay (Eco Simples com Feedback) usando Buffer Circular.
    """
    out = np.zeros_like(x)
    current_write = write_ptr
    
    for i in range(len(x)):
        # Calcula de onde ler no passado (Ponteiro de leitura)
        # Lógica circular: se for negativo, dá a volta no array
        read_ptr = current_write - delay_samps
        if read_ptr < 0:
            read_ptr += buf_len
            
        # Lê o som antigo (o eco)
        delayed_sample = buffer[read_ptr]
        
        input_val = x[i]
        
        # O som que sai é a entrada + o eco
        out[i] = input_val + delayed_sample
        
        # Escreve no buffer: Entrada atual + Eco diminuído (Feedback)
        # Isso faz o eco se repetir infinitamente caindo de volume
        buffer[current_write] = input_val + (delayed_sample * feedback)
        
        # Avança o ponteiro
        current_write += 1
        if current_write >= buf_len:
            current_write = 0
            
    return out, buffer, current_write