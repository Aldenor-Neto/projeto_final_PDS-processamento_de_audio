import tkinter as tk
from tkinter import ttk
import sounddevice as sd
import numpy as np
import queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from dsp_engine import AudioProcessor

class AccessibleAudioApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DSP Studio - Modo Acessível")
        self.geometry("1000x700")
        
        # Configuração para alto contraste (Fundo escuro, texto claro)
        self.configure(bg="#202020")
        self.style = ttk.Style()
        self.style.theme_use('default')
        
        # Cores acessíveis mas Dark
        self.style.configure(".", background="#202020", foreground="white", fieldbackground="#303030")
        self.style.configure("TLabel", background="#202020", foreground="white", font=('Arial', 12))
        self.style.configure("TButton", padding=6, font=('Arial', 11, 'bold'))
        self.style.configure("TCheckbutton", font=('Arial', 11), background="#202020", foreground="white")
        self.style.map("TCheckbutton", background=[('active', '#404040')])
        self.style.map("TButton", background=[('active', '#404040')])

        # Configurações de Áudio
        self.BLOCK_SIZE = 1024
        self.SAMPLE_RATE = 44100
        self.processor = AudioProcessor(self.BLOCK_SIZE, self.SAMPLE_RATE)
        self.stream = None
        self.is_running = False
        self.plot_queue = queue.Queue(maxsize=10)

        self._setup_ui()
        
        # Focar no botão principal ao iniciar
        self.btn_mic.focus_set()

    def _setup_ui(self):
        # Container Principal
        main_frame = tk.Frame(self, bg="#202020")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- COLUNA DA ESQUERDA (CONTROLES) ---
        # Usamos LabelFrame para agrupar semanticamente para o leitor de tela
        controls_frame = tk.LabelFrame(main_frame, text="Painel de Controle", 
                                     bg="#202020", fg="white", font=('Arial', 14, 'bold'))
        controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), ipadx=10, ipady=10)

        # Botão Microfone
        self.btn_mic = tk.Button(controls_frame, text="Ativar Microfone (Alt+M)", 
                               command=self.toggle_mic, bg="#006400", fg="white", 
                               font=('Arial', 12, 'bold'), activebackground="#008000",
                               underline=17) # Sublinha o 'M' para indicar atalho
        self.btn_mic.pack(fill=tk.X, padx=10, pady=10)
        self.bind("<Alt-m>", lambda e: self.toggle_mic())

        # Função auxiliar para criar sliders acessíveis
        def create_slider(parent, label_text, vmin, vmax, default, command, shortcut_key=None):
            frame = tk.Frame(parent, bg="#202020")
            frame.pack(fill=tk.X, padx=10, pady=5)
            
            lbl_text = label_text
            if shortcut_key:
                lbl_text += f" (Alt+{shortcut_key})"
                
            lbl = ttk.Label(frame, text=lbl_text)
            lbl.pack(anchor="w")
            
            scale = ttk.Scale(frame, from_=vmin, to=vmax, command=command)
            scale.set(default)
            scale.pack(fill=tk.X)
            
            if shortcut_key:
                self.bind(f"<Alt-{shortcut_key.lower()}>", lambda e: scale.focus_set())
                
            return scale

        # Função auxiliar para checkboxes
        def create_check(parent, text, command, shortcut_key=None):
            display_text = text
            if shortcut_key:
                display_text += f" (Alt+{shortcut_key})"
                
            var = tk.BooleanVar()
            chk = ttk.Checkbutton(parent, text=display_text, variable=var, command=command)
            chk.pack(anchor="w", padx=10, pady=2)
            
            if shortcut_key:
                self.bind(f"<Alt-{shortcut_key.lower()}>", lambda e: chk.invoke())
                
            return chk, var

        # 1. Distorção
        group_dist = tk.LabelFrame(controls_frame, text="Distorção", bg="#202020", fg="#aaaaaa")
        group_dist.pack(fill=tk.X, padx=5, pady=5)
        self.chk_dist, self.var_dist = create_check(group_dist, "Ativar Distorção", self.update_params, "D")
        self.slider_dist = create_slider(group_dist, "Ganho", 1, 50, 20, self.update_params)

        # 2. Filtros
        group_filter = tk.LabelFrame(controls_frame, text="Filtros", bg="#202020", fg="#aaaaaa")
        group_filter.pack(fill=tk.X, padx=5, pady=5)
        self.chk_high, self.var_high = create_check(group_filter, "High-Pass (300Hz)", self.update_params)
        self.chk_low, self.var_low = create_check(group_filter, "Low-Pass (5kHz)", self.update_params)

        # 3. Tremolo
        group_trem = tk.LabelFrame(controls_frame, text="Tremolo", bg="#202020", fg="#aaaaaa")
        group_trem.pack(fill=tk.X, padx=5, pady=5)
        self.chk_trem, self.var_trem = create_check(group_trem, "Ativar Tremolo", self.update_params, "T")
        self.slider_trem_rate = create_slider(group_trem, "Velocidade", 0.5, 15, 5, self.update_params)
        self.slider_trem_depth = create_slider(group_trem, "Profundidade", 0.0, 1.0, 0.6, self.update_params)

        # 4. Delay
        group_delay = tk.LabelFrame(controls_frame, text="Delay (Eco)", bg="#202020", fg="#aaaaaa")
        group_delay.pack(fill=tk.X, padx=5, pady=5)
        self.chk_delay, self.var_delay = create_check(group_delay, "Ativar Delay", self.update_params, "E") # E de Eco
        self.slider_delay_time = create_slider(group_delay, "Tempo", 0.1, 1.0, 0.4, self.update_params)
        self.slider_delay_fb = create_slider(group_delay, "Feedback", 0.0, 0.9, 0.5, self.update_params)

        # 5. Reverb
        group_reverb = tk.LabelFrame(controls_frame, text="Reverb", bg="#202020", fg="#aaaaaa")
        group_reverb.pack(fill=tk.X, padx=5, pady=5)
        self.chk_reverb, self.var_reverb = create_check(group_reverb, "Ativar Reverb", self.update_params, "R")
        self.var_reverb.set(True) # Começa ligado
        self.slider_mix = create_slider(group_reverb, "Mix Level", 0.0, 1.0, 0.3, self.update_params)

        # --- COLUNA DA DIREITA (GRÁFICOS) ---
        viz_frame = tk.Frame(main_frame, bg="#202020")
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        lbl_info = tk.Label(viz_frame, text="Visualização Gráfica (Não acessível via leitor de tela)", 
                          bg="#202020", fg="#888888", font=('Arial', 10, 'italic'))
        lbl_info.pack(pady=5)

        self.fig, (self.ax_wave, self.ax_spec) = plt.subplots(2, 1, figsize=(5, 4), facecolor='#202020')
        self.fig.subplots_adjust(hspace=0.4)
        
        # Estilização dos gráficos para combinar com o tema
        for ax in [self.ax_wave, self.ax_spec]:
            ax.set_facecolor('#101010')
            ax.tick_params(colors='white', labelsize=8)
            for spine in ax.spines.values(): spine.set_color('#404040')

        self.ax_wave.set_title("Forma de Onda", color='white', fontsize=10)
        self.ax_wave.set_ylim(-1, 1)
        self.line_wave, = self.ax_wave.plot(np.zeros(self.BLOCK_SIZE), color='#00ffcc', lw=1)

        self.ax_spec.set_title("Espectrograma FFT", color='white', fontsize=10)
        self.ax_spec.set_ylim(0, 1)
        self.ax_spec.set_xlim(0, self.SAMPLE_RATE/2)
        self.line_spec, = self.ax_spec.plot(np.zeros(self.BLOCK_SIZE // 2), color='#ff00cc', lw=1)

        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.after(50, self.update_plot)

    def update_params(self, _=None):
        # Sincroniza GUI -> Processor
        self.processor.distortion_on = self.var_dist.get()
        self.processor.tremolo_on = self.var_trem.get()
        self.processor.delay_on = self.var_delay.get()
        self.processor.reverb_on = self.var_reverb.get()
        self.processor.high_pass_on = self.var_high.get()
        self.processor.low_pass_on = self.var_low.get()

        self.processor.dist_gain = self.slider_dist.get()
        self.processor.tremolo_rate = self.slider_trem_rate.get()
        self.processor.tremolo_depth = self.slider_trem_depth.get()
        self.processor.delay_time = self.slider_delay_time.get()
        self.processor.delay_feedback = self.slider_delay_fb.get()

        wet = self.slider_mix.get()
        self.processor.mix_wet = wet
        self.processor.mix_dry = 1.0 - (wet * 0.5)

    def audio_callback(self, indata, outdata, frames, time, status):
        if status: print(status)
        x = indata[:, 0]
        y = self.processor.process_block(x)
        outdata[:, 0] = y
        outdata[:, 1] = y
        try: self.plot_queue.put_nowait(y) 
        except queue.Full: pass

    def toggle_mic(self):
        if self.is_running:
            self.stream.stop()
            self.stream.close()
            self.is_running = False
            self.btn_mic.configure(text="Ativar Microfone (Alt+M)", bg="#006400", activebackground="#008000")
            self.bell() # Som de sistema
        else:
            self.update_params()
            try:
                self.stream = sd.Stream(channels=2, samplerate=self.SAMPLE_RATE, 
                                        blocksize=self.BLOCK_SIZE, dtype='float32',
                                        callback=self.audio_callback)
                self.stream.start()
                self.is_running = True
                self.btn_mic.configure(text="PARAR MICROFONE (Alt+M)", bg="#8b0000", activebackground="#a52a2a")
                self.bell()
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
                self.canvas.draw_idle()
        except: pass
        self.after(50, self.update_plot)

if __name__ == "__main__":
    app = AccessibleAudioApp()
    app.mainloop()