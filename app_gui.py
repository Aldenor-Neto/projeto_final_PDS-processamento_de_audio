import customtkinter as ctk
import sounddevice as sd
import numpy as np
import queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from dsp_engine import AudioProcessor

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

def find_asio_device():
    """Encontra dispositivo ASIO (prioriza Roland)"""
    try:
        devices = sd.query_devices()
        asio_devices = []
        roland_device = None
        
        for i, device in enumerate(devices):
            # Verifica se é ASIO (hostapi pode ser 'asio' ou nome contém 'asio')
            hostapi_info = sd.query_hostapis(device['hostapi'])
            hostapi_name = hostapi_info['name'].lower()
            
            if 'asio' in hostapi_name:
                asio_devices.append((i, device))
                # Prioriza dispositivos Roland
                if 'roland' in device['name'].lower() or 'capture' in device['name'].lower():
                    roland_device = i
        
        # Retorna dispositivo Roland se encontrado, senão primeiro ASIO
        if roland_device is not None:
            return roland_device
        elif asio_devices:
            return asio_devices[0][0]
        else:
            return None
    except Exception as e:
        print(f"Erro ao procurar dispositivo ASIO: {e}")
        return None

class ModernAudioApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("DSP Studio - Processador em Tempo Real")
        self.geometry("1000x700")
        self.resizable(True, True)

        self.BLOCK_SIZE = 1024
        self.SAMPLE_RATE = 44100
        self.processor = AudioProcessor(self.BLOCK_SIZE, self.SAMPLE_RATE)
        self.input_stream = None
        self.output_stream = None
        self.is_running = False
        self.plot_queue = queue.Queue(maxsize=10)
        self.audio_queue = queue.Queue(maxsize=5)  # Fila para áudio processado
        self.asio_device = find_asio_device()  # Detecta dispositivo ASIO

        # Buffer para o espectrograma
        # 10 segundos * SAMPLE_RATE / BLOCK_SIZE = 10 * 44100 / 1024 = ~430.6
        self.spectrogram_buffer = np.zeros((self.BLOCK_SIZE // 2, 440)) # 440 colunas para ~10 segundos

        self._setup_ui()

    def _setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- SIDEBAR ---
        self.sidebar = ctk.CTkScrollableFrame(self, width=280, corner_radius=0, label_text="DSP RACK")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        # Botão Mic
        self.btn_mic = ctk.CTkButton(self.sidebar, text="ATIVAR MICROFONE", 
                                     command=self.toggle_mic, fg_color="#2ecc71", 
                                     hover_color="#27ae60", font=ctk.CTkFont(weight="bold"))
        self.btn_mic.pack(padx=20, pady=20, fill="x")

        # 1. Seção Distorção
        self.add_section("Distorção (Fuzz)", "distortion")
        self.slider_dist = self.add_slider("Ganho", 1, 50, 50, self.update_params)
        
        # 2. Seção Filtros
        ctk.CTkLabel(self.sidebar, text="----- Filtros -----", text_color="gray").pack(pady=10)
        self.sw_high = ctk.CTkSwitch(self.sidebar, text="High-Pass (300Hz)", command=self.update_params)
        self.sw_high.pack(padx=20, pady=5, anchor="w")
        self.sw_low = ctk.CTkSwitch(self.sidebar, text="Low-Pass (5kHz)", command=self.update_params)
        self.sw_low.pack(padx=20, pady=5, anchor="w")

        # 3. Seção Tremolo
        self.add_section("Tremolo (Modulação)", "tremolo")
        self.slider_trem_rate = self.add_slider("Velocidade (Hz)", 0.5, 15, 15, self.update_params)
        self.slider_trem_depth = self.add_slider("Profundidade", 0.0, 1.0, 1.0, self.update_params)

        # 4. Seção Delay
        self.add_section("Delay (Eco)", "delay")
        self.slider_delay_time = self.add_slider("Tempo (s)", 0.1, 1.0, 1.0, self.update_params)
        self.slider_delay_fb = self.add_slider("Feedback", 0.0, 0.9, 0.9, self.update_params)

        # 5. Seção Reverb
        self.add_section("Reverb (Convolução)", "reverb")
        self.sw_reverb.select() # Padrão ligado
        self.slider_mix = self.add_slider("Mix Level", 0.0, 1.0, 1.0, self.update_params)

        # --- ÁREA PRINCIPAL (GRÁFICOS) ---
        self.main_area = ctk.CTkFrame(self, corner_radius=10)
        self.main_area.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.lbl_viz = ctk.CTkLabel(self.main_area, text="Visualizador de Sinal", font=ctk.CTkFont(size=16))
        self.lbl_viz.pack(pady=10)

        self.fig, (self.ax_wave, self.ax_spec, self.ax_phase) = plt.subplots(3, 1, figsize=(5, 6), facecolor='#2b2b2b') # Aumentado para 3 subplots
        self.fig.subplots_adjust(hspace=0.6) # Ajustado o espaçamento vertical

        self.ax_wave.set_title("Osciloscópio", color='white', fontsize=9)
        self.ax_wave.set_ylim(-1, 1)
        self.ax_wave.set_facecolor('#1a1a1a')
        self.ax_wave.tick_params(colors='white', labelsize=8)
        self.line_wave, = self.ax_wave.plot(np.zeros(self.BLOCK_SIZE), color='#00ffcc', lw=1)

        self.ax_spec.set_title("Espectrograma FFT", color='white', fontsize=9)
        self.ax_spec.set_ylim(0, 1)
        self.ax_spec.set_xlim(0, self.SAMPLE_RATE/2)
        self.ax_spec.set_facecolor('#1a1a1a')
        self.ax_spec.tick_params(colors='white', labelsize=8)
        self.line_spec, = self.ax_spec.plot(np.zeros(self.BLOCK_SIZE // 2), color='#ff00cc', lw=1)

        self.ax_phase.set_title("Espectrograma de Frequência x Tempo", color='white', fontsize=9)
        self.ax_phase.set_xlabel("Tempo (s)", color='white', fontsize=8) # Rótulo do eixo X para segundos
        self.ax_phase.set_ylabel("Frequência (Hz)", color='white', fontsize=8)
        self.ax_phase.set_facecolor('#1a1a1a')
        self.ax_phase.tick_params(colors='white', labelsize=8)
        self.ax_phase.set_ylim(0, 10000) # Frequência ajustada para 10000 Hz
        self.ax_phase.set_xlim(0, 10) # Tempo ajustado para 0 a 10 segundos

        # Usar imshow para o espectrograma
        self.im_spectrogram = self.ax_phase.imshow(self.spectrogram_buffer, 
                                                    origin='lower', aspect='auto', 
                                                    cmap='inferno', 
                                                    extent=[0, 10, 0, self.SAMPLE_RATE/2],
                                                    vmin=-60, vmax=0) # Ajustado vmin e vmax para escala em dB

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_area)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        self.after(50, self.update_plot)

    def add_section(self, title, attr_prefix):
        """Helper para criar seções UI"""
        ctk.CTkLabel(self.sidebar, text=f"----- {title} -----", text_color="gray").pack(pady=(15, 5))
        switch = ctk.CTkSwitch(self.sidebar, text="Ativar", command=self.update_params)
        switch.pack(padx=20, pady=5, anchor="w")
        # Salva referencia do switch dinamicamente (ex: self.sw_distortion)
        setattr(self, f"sw_{attr_prefix}", switch)

    def add_slider(self, text, vmin, vmax, vdefault, command):
        ctk.CTkLabel(self.sidebar, text=text, font=ctk.CTkFont(size=11)).pack(padx=20, pady=(5,0), anchor="w")
        slider = ctk.CTkSlider(self.sidebar, from_=vmin, to=vmax, command=command, height=15)
        slider.set(vdefault)
        slider.pack(padx=20, pady=5, fill="x")
        return slider

    def update_params(self, _=None):
        # Switches
        self.processor.distortion_on = bool(self.sw_distortion.get())
        self.processor.tremolo_on = bool(self.sw_tremolo.get())
        self.processor.delay_on = bool(self.sw_delay.get())
        self.processor.reverb_on = bool(self.sw_reverb.get())
        self.processor.high_pass_on = bool(self.sw_high.get())
        self.processor.low_pass_on = bool(self.sw_low.get())

        # Sliders
        self.processor.dist_gain = self.slider_dist.get()
        
        self.processor.tremolo_rate = self.slider_trem_rate.get()
        self.processor.tremolo_depth = self.slider_trem_depth.get()
        
        self.processor.delay_time = self.slider_delay_time.get()
        self.processor.delay_feedback = self.slider_delay_fb.get()

        wet = self.slider_mix.get()
        self.processor.mix_wet = wet
        self.processor.mix_dry = 1.0 - (wet * 0.5)

    def input_callback(self, indata, frames, time, status):
        """Callback para captura de áudio - apenas processa, não reproduz"""
        if status: print(status)
        x = indata[:, 0]
        y = self.processor.process_block(x)
        try: 
            self.audio_queue.put_nowait(y)
            self.plot_queue.put_nowait(y)
        except queue.Full: 
            pass
    
    def output_callback(self, outdata, frames, time, status):
        """Callback para reprodução - apenas reproduz o áudio processado"""
        if status: print(status)
        try:
            # Pega o áudio processado da fila
            audio_data = self.audio_queue.get_nowait()
            outdata[:, 0] = audio_data
            outdata[:, 1] = audio_data
        except queue.Empty:
            # Se não houver áudio processado, silencia
            outdata.fill(0)

    def toggle_mic(self):
        if self.is_running:
            self.input_stream.stop()
            self.input_stream.close()
            self.output_stream.stop()
            self.output_stream.close()
            self.is_running = False
            # Limpa as filas
            while not self.audio_queue.empty():
                try: self.audio_queue.get_nowait()
                except: break
            self.btn_mic.configure(text="ATIVAR MICROFONE", fg_color="#2ecc71")
        else:
            self.update_params()
            # Configura dispositivo ASIO se disponível
            device = self.asio_device
            if device is not None:
                print(f"Usando dispositivo ASIO: {sd.query_devices(device)['name']}")
            
            try:
                # Streams separados: entrada e saída independentes
                # Usa ASIO para evitar monitoramento automático e reduzir latência
                stream_kwargs = {
                    'samplerate': self.SAMPLE_RATE,
                    'blocksize': self.BLOCK_SIZE,
                    'dtype': 'float32',
                    'latency': 'low'  # Latência mínima com ASIO
                }
                
                if device is not None:
                    stream_kwargs['device'] = device
                
                self.input_stream = sd.InputStream(
                    channels=1,
                    callback=self.input_callback,
                    **stream_kwargs
                )
                self.output_stream = sd.OutputStream(
                    channels=2,
                    callback=self.output_callback,
                    **stream_kwargs
                )
                self.input_stream.start()
                self.output_stream.start()
                self.is_running = True
                self.btn_mic.configure(text="PARAR", fg_color="#c0392b")
            except Exception as e:
                print(f"Erro ao iniciar streams com ASIO: {e}")
                # Tenta sem especificar dispositivo se ASIO falhar
                try:
                    print("Tentando sem dispositivo ASIO específico...")
                    self.input_stream = sd.InputStream(
                        channels=1,
                        samplerate=self.SAMPLE_RATE,
                        blocksize=self.BLOCK_SIZE,
                        dtype='float32',
                        callback=self.input_callback,
                        latency='low'
                    )
                    self.output_stream = sd.OutputStream(
                        channels=2,
                        samplerate=self.SAMPLE_RATE,
                        blocksize=self.BLOCK_SIZE,
                        dtype='float32',
                        callback=self.output_callback,
                        latency='low'
                    )
                    self.input_stream.start()
                    self.output_stream.start()
                    self.is_running = True
                    self.btn_mic.configure(text="PARAR", fg_color="#c0392b")
                except Exception as e2:
                    print(f"Erro ao iniciar streams sem ASIO: {e2}")

    def update_plot(self):
        try:
            while not self.plot_queue.empty():
                data = self.plot_queue.get_nowait()
                self.line_wave.set_ydata(data)
                self.line_wave.set_xdata(np.arange(len(data)))
                
                fft_data = np.fft.fft(data)
                mag = np.abs(fft_data[:len(data)//2])
                mag = mag * 4 / self.BLOCK_SIZE
                
                # Converte para decibéis
                mag_db = 20 * np.log10(mag + 1e-10) # Adiciona um pequeno offset para evitar log(0)

                self.line_spec.set_ydata(mag)
                self.line_spec.set_xdata(np.linspace(0, self.SAMPLE_RATE/2, len(mag)))

                # Atualiza o buffer do espectrograma com valores em dB
                self.spectrogram_buffer = np.roll(self.spectrogram_buffer, -1, axis=1)
                self.spectrogram_buffer[:, -1] = mag_db
                self.im_spectrogram.set_array(self.spectrogram_buffer)

                self.canvas.draw_idle()
        except: pass
        self.after(50, self.update_plot)

if __name__ == "__main__":
    app = ModernAudioApp()
    app.mainloop()