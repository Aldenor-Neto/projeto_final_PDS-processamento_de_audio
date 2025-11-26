import customtkinter as ctk
import sounddevice as sd
import numpy as np
import queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from dsp_engine import AudioProcessor
from collections import deque # Adicionar esta importação

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

class ModernAudioApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("DSP Studio - Processador em Tempo Real")
        self.geometry("1000x700")
        self.resizable(True, True)

        self.BLOCK_SIZE = 1024
        self.SAMPLE_RATE = 44100
        self.processor = AudioProcessor(self.BLOCK_SIZE, self.SAMPLE_RATE)
        self.stream = None
        self.is_running = False
        self.plot_queue = queue.Queue(maxsize=10)
        
        self.SPECTROGRAM_HISTORY_SIZE = 100 # Nova constante para o tamanho do histórico
        self.spec_history = deque(maxlen=self.SPECTROGRAM_HISTORY_SIZE) # Histórico para o espectrograma de tempo e frequência

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
        self.slider_dist = self.add_slider("Ganho", 1, 50, 20, self.update_params)
        
        # 2. Seção Filtros
        ctk.CTkLabel(self.sidebar, text="----- Filtros -----", text_color="gray").pack(pady=10)
        self.sw_high = ctk.CTkSwitch(self.sidebar, text="High-Pass (300Hz)", command=self.update_params)
        self.sw_high.pack(padx=20, pady=5, anchor="w")
        self.sw_low = ctk.CTkSwitch(self.sidebar, text="Low-Pass (5kHz)", command=self.update_params)
        self.sw_low.pack(padx=20, pady=5, anchor="w")

        # 3. Seção Tremolo
        self.add_section("Tremolo (Modulação)", "tremolo")
        self.slider_trem_rate = self.add_slider("Velocidade (Hz)", 0.5, 15, 5, self.update_params)
        self.slider_trem_depth = self.add_slider("Profundidade", 0.0, 1.0, 0.6, self.update_params)

        # 4. Seção Delay
        self.add_section("Delay (Eco)", "delay")
        self.slider_delay_time = self.add_slider("Tempo (s)", 0.1, 1.0, 0.4, self.update_params)
        self.slider_delay_fb = self.add_slider("Feedback", 0.0, 0.9, 0.5, self.update_params)

        # 5. Seção Reverb
        self.add_section("Reverb (Convolução)", "reverb")
        self.sw_reverb.select() # Padrão ligado
        self.slider_mix = self.add_slider("Mix Level", 0.0, 1.0, 0.3, self.update_params)

        # --- ÁREA PRINCIPAL (GRÁFICOS) ---
        self.main_area = ctk.CTkFrame(self, corner_radius=10)
        self.main_area.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.lbl_viz = ctk.CTkLabel(self.main_area, text="Visualizador de Sinal", font=ctk.CTkFont(size=16))
        self.lbl_viz.pack(pady=10)

        self.fig, (self.ax_wave, self.ax_spec, self.ax_time_freq) = plt.subplots(3, 1, figsize=(5, 6), facecolor='#2b2b2b') # Alterar para 3 subplots
        self.fig.subplots_adjust(hspace=0.6) # Ajustar espaçamento

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

        # Novo gráfico: Espectrograma de tempo e frequência
        self.ax_time_freq.set_title("Intensidade da Frequência (Tempo)", color='white', fontsize=9)
        self.ax_time_freq.set_ylabel("Frequência (Hz)", color='white', fontsize=8)
        self.ax_time_freq.set_xlabel("Tempo", color='white', fontsize=8)
        self.ax_time_freq.set_facecolor('#1a1a1a') # Definir a cor de fundo
        self.ax_time_freq.tick_params(colors='white', labelsize=8)
        self.ax_time_freq.set_ylim(0, self.SAMPLE_RATE / 2)
        self.ax_time_freq.set_xlim(0, self.SPECTROGRAM_HISTORY_SIZE)

        # Inicializa o imshow com dados vazios
        self.img_time_freq = self.ax_time_freq.imshow(
            np.zeros((self.BLOCK_SIZE // 2, self.SPECTROGRAM_HISTORY_SIZE)),
            origin='lower',
            aspect='auto',
            cmap='hot', # Alterei de 'magma' para 'hot'
            extent=[0, self.SPECTROGRAM_HISTORY_SIZE, 0, self.SAMPLE_RATE / 2],
            vmin=0.0001, # Ajustei vmin para um valor ainda menor
            vmax=0.02, # Ajustei vmax para um valor ainda menor
            interpolation='bilinear'
        )

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
        self.processor.mix_dry = 1.0 - wet # Alterado de 1.0 - (wet * 0.5)

    def audio_callback(self, indata, outdata, frames, time, status):
        if status: print(status)
        x = indata[:, 0]
        y = self.processor.process_block(x)
        outdata[:, 0] = y
        try: self.plot_queue.put_nowait(y) 
        except queue.Full: pass

    def toggle_mic(self):
        if self.is_running:
            self.stream.stop()
            self.stream.close()
            self.is_running = False
            self.btn_mic.configure(text="ATIVAR MICROFONE", fg_color="#2ecc71")
        else:
            self.update_params()
            try:
                self.stream = sd.Stream(channels=1, samplerate=self.SAMPLE_RATE, 
                                        blocksize=self.BLOCK_SIZE, dtype='float32',
                                        callback=self.audio_callback)
                self.stream.start()
                self.is_running = True
                self.btn_mic.configure(text="PARAR", fg_color="#c0392b")
            except Exception as e:
                print(e)

    def update_plot(self):
        try:
            while not self.plot_queue.empty():
                data = self.plot_queue.get_nowait()
                self.line_wave.set_ydata(data)
                self.line_wave.set_xdata(np.arange(len(data)))
                
                fft_data = np.fft.fft(data)
                mag = np.abs(fft_data[:len(data)//2])
                mag = mag * 4 / self.BLOCK_SIZE
                self.line_spec.set_ydata(mag)
                self.line_spec.set_xdata(np.linspace(0, self.SAMPLE_RATE/2, len(mag)))
                
                # Adicionar os dados ao histórico do espectrograma de tempo e frequência
                self.spec_history.append(mag)
                if len(self.spec_history) > self.SPECTROGRAM_HISTORY_SIZE:
                    self.spec_history.popleft()

                # Atualizar o gráfico de intensidade da frequência ao longo do tempo
                # Transpor a matriz para que o tempo seja o eixo X e a frequência o eixo Y
                spec_data_2d = np.array(list(self.spec_history)).T
                self.img_time_freq.set_array(spec_data_2d)
                
                self.canvas.draw_idle()
        except: pass
        self.after(50, self.update_plot)

if __name__ == "__main__":
    app = ModernAudioApp()
    app.mainloop()